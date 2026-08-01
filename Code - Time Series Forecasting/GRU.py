import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体以支持中文

# 生成时间序列数据
time_steps = np.linspace(0, 100, 1000)
data = np.sin(time_steps) + np.random.normal(0, 0.05, len(time_steps))  # 添加了一些较小的噪声

# 利用GRU进行时间序列预测
def GRU_time_series_predict(data,time_steps,time_steps_to_predict):

    '''
    - 输入1: data: 原始的时间序列数据，以一维数组的形式输入，比如np.array([1.2, 2.1, 3.2])
    - 输入2: time_steps: data各个数据对应的时间，长度与data一致，比如np.array([1, 2, 3])
    - 输入3: time_steps_to_predict: 要预测的时间点，可以是任意长度的列表，比如[4, 5]
    如果拟合优度等指标不佳，调整代码中的dropout率、迭代次数等
    '''
    
    # ADF检验函数
    def adf_test(timeseries):
        adf_result = adfuller(timeseries)
        print(f'ADF Statistic: {adf_result[0]}')
        print(f'p-value: {adf_result[1]}')
        for key, value in adf_result[4].items():
            print(f'Critical Values: {key}, {value}')
        return adf_result[1]

    # 进行ADF检验
    print("原始数据的ADF检验:")
    p_value = adf_test(data)

    # 如果p-value大于0.05，数据非平稳，需要差分
    if p_value > 0.05:
        data_diff = np.diff(data, n=1)  # 一阶差分
        print("\n差分后的数据ADF检验:")
        p_value_diff = adf_test(data_diff)
    else:
        data_diff = data

    # 归一化数据到0和1之间
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_diff = scaler.fit_transform(data_diff.reshape(-1, 1))

    # 创建数据集矩阵
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    # 选择时间步
    time_step = 30  # 增加 time_step 来捕捉更多的历史信息
    X, Y = create_dataset(data_diff, time_step)

    # 转换为PyTorch的Tensor格式
    X = torch.tensor(X, dtype=torch.float32).reshape(-1, time_step, 1)
    Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    # 定义GRU模型
    class GRUModel(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=150, output_size=1):
            super(GRUModel, self).__init__()
            self.hidden_layer_size = hidden_layer_size
            self.gru = nn.GRU(input_size, hidden_layer_size, num_layers=4, batch_first=True)  # 使用GRU
            self.dropout = nn.Dropout(0.3)
            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, x):
            h_0 = torch.zeros(4, x.size(0), self.hidden_layer_size)  # 4层GRU的隐藏状态

            gru_out, _ = self.gru(x, h_0)
            gru_out = self.dropout(gru_out)
            predictions = self.linear(gru_out[:, -1, :])
            return predictions

    # 初始化模型、损失函数和优化器
    model = GRUModel()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)  # 使用RMSprop优化器

    # 用于记录损失值
    losses = []

    # 训练模型
    num_epochs = 100  # 增加训练的epoch数量
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()

        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'第 {epoch+1}/{num_epochs} 次迭代, 损失: {loss.item():.4f}')

    # 进行预测
    model.eval()
    predictions = model(X).detach().numpy()

    # 反归一化数据
    predictions = scaler.inverse_transform(predictions)

    # 如果进行了差分，需要反差分以恢复原始数据
    if p_value > 0.05:
        predictions = np.cumsum(predictions) + data[time_step]  # 反差分

    # 确保 predictions 的长度与 data[time_step:] 一致
    if len(predictions) < len(data[time_step + 1:]):
        predictions = np.concatenate([np.array([data[time_step]]), predictions])

    # 计算评价指标
    mse = mean_squared_error(data[time_step + 1:], predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(data[time_step + 1:], predictions)
    r2 = r2_score(data[time_step + 1:], predictions)

    print(f'均方误差 (MSE): {mse:.4f}')
    print(f'均方根误差 (RMSE): {rmse:.4f}')
    print(f'平均绝对误差 (MAE): {mae:.4f}')
    print(f'决定系数 (R²): {r2:.4f}')

    # 绘制预测结果与实际结果的拟合曲线
    plt.figure(figsize=(14,8))
    plt.plot(time_steps[time_step + 1:], data[time_step + 1:], label='实际数据')
    plt.plot(time_steps[time_step + 1:], predictions, label='预测数据')
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.title('实际数据与预测数据对比')
    plt.legend()
    plt.show()

    # 绘制损失函数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), losses, label='训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('损失函数曲线')
    plt.legend()
    plt.show()

    # 定义函数接收时间步并输出预测结果
    def predict_for_time_steps(time_steps_to_predict):
        # 创建新的数据集矩阵以供预测
        X_predict = []
        for i in time_steps_to_predict:
            if i < time_step:
                raise ValueError("输入的时间步长度小于模型要求的 time_step.")
            X_predict.append(data_diff[i-time_step:i, 0])
        
        X_predict = torch.tensor(np.array(X_predict), dtype=torch.float32).reshape(-1, time_step, 1)
        
        # 进行预测
        model.eval()
        predictions = model(X_predict).detach().numpy()
        
        # 反归一化数据
        predictions = scaler.inverse_transform(predictions)
        
        # 如果进行了差分，需要反差分以恢复原始数据
        if p_value > 0.05:
            predictions = np.cumsum(predictions) + data[time_steps_to_predict[0] - time_step]
        
        return predictions

    # 使用函数预测给定的时间步
    predicted_values = predict_for_time_steps(time_steps_to_predict)
    predicted_values = np.array(predicted_values).flatten().tolist()

    return predicted_values


print("GRU时间序列预测的结果是:", GRU_time_series_predict(data, time_steps, [100, 101, 102]))
