import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 生成虚拟数据集
np.random.seed(42)
X1 = np.random.rand(1000, 1) * 10
X2 = np.random.rand(1000, 1) * 10
y = np.sin(X1).ravel() + np.cos(X2).ravel() + np.random.normal(0, 0.5, X1.shape[0])

# 合并特征
X = np.hstack((X1, X2))

# 生成多项式特征
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 训练随机森林回归模型
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# 预测
y_pred = rf.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'均方误差 (MSE): {mse:.4f}')
print(f'均方根误差 (RMSE): {rmse:.4f}')
print(f'平均绝对误差 (MAE): {mae:.4f}')
print(f'拟合优度 (R^2): {r2:.4f}')

# 创建 DataFrame 以便绘图
df = pd.DataFrame({'X1': X_test[:, 0], 'X2': X_test[:, 1], 'y_true': y_test, 'y_pred': y_pred})

# 绘制实际值 vs 预测值
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('实际值 vs 预测值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.grid(True)

# 绘制残差图
plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_test, residuals, color='purple', alpha=0.5)
plt.hlines(0, xmin=y_test.min(), xmax=y_test.max(), colors='gray', linestyles='dashed')
plt.title('残差图')
plt.xlabel('实际值')
plt.ylabel('残差')
plt.grid(True)
plt.tight_layout()
plt.show()

# 特征重要性
feature_importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances, tick_label=[f'特征 {i+1}' for i in range(X_train.shape[1])])
plt.title('特征重要性')
plt.show()

# 绘制预测 vs 实际值的密度图
plt.figure(figsize=(8, 6))
sns.kdeplot(y_test, label='实际值', color='blue', fill=True, alpha=0.3)
sns.kdeplot(y_pred, label='预测值', color='red', fill=True, alpha=0.3)
plt.title('实际值 vs 预测值的密度图')
plt.xlabel('y')
plt.ylabel('密度')
plt.legend()
plt.show()

# 使用热图显示预测误差分布
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('预测误差分布热图')
plt.show()

# 用户输入特征值进行预测
input_X1 = float(input("请输入特征值X1（范围在0到10之间）: "))
input_X2 = float(input("请输入特征值X2（范围在0到10之间）: "))
input_features = poly.transform([[input_X1, input_X2]])
predicted_value = rf.predict(input_features)
print(f'输入的特征值X1: {input_X1} 和 X2: {input_X2} 对应的预测值y: {predicted_value[0]}')
