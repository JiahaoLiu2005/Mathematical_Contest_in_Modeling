import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建虚拟数据集
np.random.seed(42)
X = np.sort(5 * np.random.rand(1000, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 拟合SVR模型
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X, y)
y_pred = svr_rbf.predict(X)

# 计算残差
residuals = y - y_pred

# 绘制原始数据和回归曲线
plt.figure(figsize=(14, 6))

plt.subplot(2, 2, 1)
plt.scatter(X, y, color='darkorange', label='数据')
plt.plot(X, y_pred, color='navy', lw=2, label='SVR 模型')
plt.xlabel('数据')
plt.ylabel('目标')
plt.title('SVR 回归')
plt.legend()

# 绘制残差图
plt.subplot(2, 2, 2)
plt.scatter(X, residuals, color='red', edgecolor='w', label='残差')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('数据')
plt.ylabel('残差')
plt.title('SVR 残差')

# 绘制支持向量
plt.subplot(2, 2, 3)
plt.scatter(X, y, color='darkorange', label='数据')
plt.scatter(X[svr_rbf.support_], y[svr_rbf.support_], facecolors='none', edgecolors='k', 
            s=100, label='支持向量')
plt.xlabel('数据')
plt.ylabel('目标')
plt.title('支持向量')
plt.legend()

# 绘制残差直方图
plt.subplot(2, 2, 4)
plt.hist(residuals, bins=20, color='blue', edgecolor='black')
plt.xlabel('残差')
plt.ylabel('频率')
plt.title('残差直方图')

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
input_value = float(input("请输入特征值X（范围在0到5之间）: "))
predicted_value = svr_rbf.predict([[input_value]])
print(f'输入的特征值X: {input_value} 对应的预测值y: {predicted_value[0]}')
