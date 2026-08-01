import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 生成虚拟数据
np.random.seed(42)
n_samples = 1000
temperature = np.random.uniform(20, 1000, n_samples)
time = np.random.uniform(1, 10, n_samples)
reaction_speed = 0.05 * temperature**1.5 - 0.5 * time**2 + np.random.normal(0, 5, n_samples)

# 创建DataFrame
data = pd.DataFrame({'Temperature': temperature, 'Time': time, 'ReactionSpeed': reaction_speed})

# 特征和目标
X = data[['Temperature', 'Time']]
y = data['ReactionSpeed']

# 决策树回归模型
model = DecisionTreeRegressor(max_depth=5)
model.fit(X, y)

# 预测
X_test_temp = np.linspace(20, 100, 100)
X_test_time = np.linspace(1, 10, 100)
X_test_temp, X_test_time = np.meshgrid(X_test_temp, X_test_time)
X_test = pd.DataFrame({'Temperature': X_test_temp.ravel(), 'Time': X_test_time.ravel()})
Z_pred = model.predict(X_test).reshape(X_test_temp.shape)

# 图形1: 3D 表面图
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X_test_temp, X_test_time, Z_pred, cmap='viridis', edgecolor='none')
ax.set_xlabel('温度')
ax.set_ylabel('时间')
ax.set_zlabel('预测反应速度')
ax.set_title('预测反应速度的3D表面图')

# 图形2: 预测 vs 实际
ax2 = fig.add_subplot(122)
y_pred = model.predict(X)
ax2.scatter(y, y_pred, c='blue', marker='o', edgecolor='w', s=70)
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax2.set_xlabel('实际反应速度')
ax2.set_ylabel('预测反应速度')
ax2.set_title('预测值 vs 实际值')

plt.tight_layout()
plt.show()

# 计算评价指标
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'均方误差 (MSE): {mse}')
print(f'均方根误差 (RMSE): {rmse}')
print(f'平均绝对误差 (MAE): {mae}')
print(f'拟合优度 (R^2): {r2}')

# 用户输入特征值进行预测
input_temp = float(input("请输入温度值（范围在20到1000之间）: "))
input_time = float(input("请输入时间值（范围在1到10之间）: "))
input_data = pd.DataFrame({'Temperature': [input_temp], 'Time': [input_time]})
predicted_speed = model.predict(input_data)
print(f'输入的温度值: {input_temp}，时间值: {input_time} 对应的预测反应速度: {predicted_speed[0]}')
