import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# 设置随机种子，保证结果可重复
np.random.seed(42)

# 生成样本数量和特征数量
num_samples = 1000
num_features = 10

# 生成特征数据 X
X = np.random.rand(num_samples, num_features)

# 生成目标变量 y（假设为二分类问题）
y = np.random.randint(2, size=num_samples)

# 将特征数据转换成 DataFrame，命名列
feature_names = [f'Feature_{i+1}' for i in range(num_features)]
X_df = pd.DataFrame(X, columns=feature_names)

# 将目标变量转换成 Series
y_series = pd.Series(y, name='Target')

# 合并特征和目标变量成一个 DataFrame
df = pd.concat([X_df, y_series], axis=1)

# 划分特征和目标变量
X = df.drop(['Target'], axis=1)
y = df['Target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 查看DataFrame的前五行数据
print(df.head())

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归模型，使用L1正则化
log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
log_reg.fit(X_train_scaled, y_train)

# 打印逻辑回归模型的系数
coefficients = pd.Series(log_reg.coef_[0], index=X.columns)
selected_features = coefficients[coefficients != 0].index
print("Selected features:")
print(selected_features)

# 预测测试集
y_pred = log_reg.predict(X_test_scaled)
y_pred_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# 计算评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评价指标
print(f"\nAccuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# 绘制ROC曲线和计算AUC值
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 定义函数进行预测
def predict(input_features):
    input_df = pd.DataFrame([input_features], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    return log_reg.predict(input_scaled)[0]

# 示例：输入要预测的特征
input_features = np.random.rand(num_features)
predicted_label = predict(input_features)
print(f"\nInput Features: {input_features}")
print(f"Predicted Label: {predicted_label}")