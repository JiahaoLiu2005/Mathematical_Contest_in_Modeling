import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 生成随机数据，直接用 numpy
np.random.seed(0)
X = np.random.rand(1000, 4)  # 生成1000个样本，每个样本有4个特征
Y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * X[:, 2] + 0.1 * X[:, 3] + np.random.normal(0, 0.1, 1000)  # 生成目标变量，带有噪声

# 转换为Tensor
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# 定义一个改进的全连接神经网络，使用较低的Dropout率
class ImprovedNN(nn.Module):
    def __init__(self):
        super(ImprovedNN, self).__init__()
        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(4, 20)
        
        # 第一个隐藏层到第二个隐藏层
        self.fc2 = nn.Linear(20, 10)
        self.dropout = nn.Dropout(0.1)  # Dropout 层，丢弃 10% 的神经元
        
        # 第二个隐藏层到输出层
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # 使用 Dropout
        x = self.fc3(x)
        return x

# 初始化改进的神经网络、损失函数和优化器
model = ImprovedNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练神经网络（记录损失函数值）
batch_size = 32
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    permutation = torch.randperm(X.size()[0])

    for i in range(0, X.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices], Y[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # 记录损失值
    losses.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型评估
with torch.no_grad():
    predicted = model(X).numpy()
    actual = Y.numpy()

# 计算评价指标
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'R²: {r2:.4f}')

# 拟合曲线
plt.figure(figsize=(10, 6))
plt.scatter(actual, predicted, c='blue', marker='o', label='Predicted vs Actual')
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Predicted vs Actual')
plt.legend()
plt.show()

# 损失函数变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function over Epochs')
plt.legend()
plt.show()

# 残差分析图
residuals = actual - predicted
plt.figure(figsize=(10, 6))
plt.scatter(predicted, residuals, c='blue', marker='o', label='Residuals')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals Analysis')
plt.legend()
plt.show()

# 假设我们要预测的新特征值如下
new_data = np.array([[0.6, 0.7, 0.8, 0.9]])

# 转换为 Tensor
new_data = torch.tensor(new_data, dtype=torch.float32)

# 使用训练好的模型进行预测
with torch.no_grad():
    predicted = model(new_data)
    print(f'Predicted value: {predicted.item():.4f}')