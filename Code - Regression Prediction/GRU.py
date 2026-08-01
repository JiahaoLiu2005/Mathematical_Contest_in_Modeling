import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 设置中文字体和解决负号显示问题
plt.rcParams['font.family'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成示例数据
numTimeSteps = 200
numFeatures = 4
numResponses = 1
X = np.random.rand(numTimeSteps, numFeatures)
Y = np.sum(X, axis=1).reshape(-1, 1) + 0.1 * np.random.randn(numTimeSteps, numResponses)

# 检查数据是否平稳
def check_stationarity(data):
    result = adfuller(data)
    return result[1] < 0.05, result[1]

is_stationary, p_value = check_stationarity(Y.flatten())
print(f'原始数据ADF检验的p值: {p_value:.4f}')

if not is_stationary:
    # 数据不平稳，进行差分处理
    Y_diff = np.diff(Y, axis=0)
    X = X[1:, :]
else:
    Y_diff = Y

# 将数据拆分为训练集和测试集
trainRatio = 0.8
numTrain = int(trainRatio * len(Y_diff))
XTrain = X[:numTrain, :]
YTrain = Y_diff[:numTrain, :]
XTest = X[numTrain:, :]
YTest = Y_diff[numTrain:, :]

# 归一化数据
scaler_X = StandardScaler()
XTrain = scaler_X.fit_transform(XTrain)
XTest = scaler_X.transform(XTest)

scaler_Y = StandardScaler()
YTrain = scaler_Y.fit_transform(YTrain)

# 定义GRU网络
class GRURegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(GRURegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 参数设定
input_size = 4
hidden_size = 200
output_size = 1
num_layers = 3
num_epochs = 1000
learning_rate = 0.001

# 创建模型
model = GRURegression(input_size, hidden_size, output_size, num_layers).to('cpu')

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 将数据调整为GRU的输入形状 [批次大小, 时间步长, 输入特征]
XTrain = torch.tensor(XTrain, dtype=torch.float32).unsqueeze(1).to('cpu')
YTrain = torch.tensor(YTrain, dtype=torch.float32).to('cpu')
XTest = torch.tensor(XTest, dtype=torch.float32).unsqueeze(1).to('cpu')
YTest = torch.tensor(YTest, dtype=torch.float32).to('cpu')

# 用于记录损失值的列表
train_losses = []

# 训练模型
for epoch in range(num_epochs):
    model.train()
    outputs = model(XTrain)
    loss = criterion(outputs, YTrain)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch+1) % 50 == 0:
        print(f'训练轮次 [{epoch+1}/{num_epochs}], 损失: {loss.item():.4f}')

# 绘制训练损失曲线
plt.figure()
plt.plot(train_losses, label='训练损失')
plt.xlabel('训练轮次', fontproperties='SimHei')
plt.ylabel('损失', fontproperties='SimHei')
plt.title('训练损失曲线', fontproperties='SimHei')
plt.grid(True)
plt.legend()
plt.show()

# 测试模型
model.eval()
with torch.no_grad():
    predicted = model(XTest).cpu().numpy()
    predicted = scaler_Y.inverse_transform(predicted)  # 反归一化

    if not is_stationary:
        # 反差分并恢复原始尺度
        Y_pred = Y[numTrain-1] + np.cumsum(predicted, axis=0)
    else:
        Y_pred = predicted

    # 确保 YTest 和 Y_pred 长度一致
    YTest = YTest.cpu().numpy()
    Y_pred = Y_pred[:len(YTest)]

    # 计算评估指标
    r2 = r2_score(YTest, Y_pred)
    mae = mean_absolute_error(YTest, Y_pred)
    mse = mean_squared_error(YTest, Y_pred)
    rmse = np.sqrt(mse)

    print(f'测试数据上的R方: {r2:.4f}')
    print(f'测试数据上的MAE: {mae:.4f}')
    print(f'测试数据上的MSE: {mse:.4f}')
    print(f'测试数据上的RMSE: {rmse:.4f}')

# 可视化预测结果与实际值对比
plt.figure()
plt.plot(np.arange(len(YTest)), YTest, 'b-', label='实际值')
plt.plot(np.arange(len(YTest)), Y_pred, 'r--', label='预测值')
plt.legend(prop={'family': 'SimHei'})
plt.title('GRU预测结果与实际值对比', fontproperties='SimHei')
plt.xlabel('时间步', fontproperties='SimHei')
plt.ylabel('响应值', fontproperties='SimHei')
plt.grid(True)
plt.show()

# 绘制残差分析图
residuals = YTest - Y_pred
plt.figure()
plt.plot(np.arange(len(residuals)), residuals, 'g-', label='残差')
plt.axhline(y=0, color='r', linestyle='--')
plt.legend(prop={'family': 'SimHei'})
plt.title('残差分析图', fontproperties='SimHei')
plt.xlabel('时间步', fontproperties='SimHei')
plt.ylabel('残差', fontproperties='SimHei')
plt.grid(True)
plt.show()
