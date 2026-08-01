import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_curve, auc

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 手动输入特征值
X1 = [0.5, 1.2, 2.3, 3.1, 4.6, 5.0, 1.1, 2.2, 3.3, 4.0, 0.6, 1.7, 2.4, 3.9, 4.7, 5.3, 1.4, 2.8, 3.2, 4.1]
X2 = [2.4, 3.3, 4.1, 5.6, 1.7, 2.9, 3.4, 4.6, 5.2, 1.9, 2.1, 3.5, 4.0, 5.3, 1.4, 2.8, 3.2, 4.9, 5.6, 1.5]
X = np.column_stack((X1, X2))  # 合并为特征矩阵

# 手动输入标签值
Y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# 使用随机森林进行分类
RFModel = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
RFModel.fit(X, Y)

# 使用原始数据进行预测
Ypred = RFModel.predict(X)

# 计算和显示混淆矩阵
confMat = confusion_matrix(Y, Ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=['类别 1', '类别 2'])
disp.plot(cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

# 计算精确率、召回率和F1评分
precision = precision_score(Y, Ypred, average='macro', zero_division=0)
recall = recall_score(Y, Ypred, average='macro', zero_division=0)
f1 = f1_score(Y, Ypred, average='macro', zero_division=0)

print(f'精确率: {precision:.2f}')
print(f'召回率: {recall:.2f}')
print(f'F1评分: {f1:.2f}')

# 绘制ROC曲线
Y_score = RFModel.predict_proba(X)[:, 1]  # 获取类别2的预测概率
fpr, tpr, _ = roc_curve(Y, Y_score, pos_label=2)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真正率')
plt.title('接收者操作特性 (ROC) 曲线')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 可视化数据和决策边界
plt.figure()
# 可视化训练数据
plt.scatter(X1[:10], X2[:10], c='red', marker='o', label='类别 1')
plt.scatter(X1[10:], X2[10:], c='blue', marker='s', label='类别 2')

# 绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = RFModel.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

plt.title('随机森林分类')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.legend(loc='best')
plt.grid(True)
plt.show()
