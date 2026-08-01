import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.rcParams['font.family'] = 'simHei' # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 生成一些分类数据
np.random.seed(0)  # 保证结果可重复
X = np.vstack([np.random.rand(10, 2) * 5, np.random.rand(10, 2) * 5 + 1])  # 20个样本, 2个特征
Y = np.hstack([np.ones(10), 2 * np.ones(10)])  # 两类标签

# 使用决策树进行分类
DTModel = DecisionTreeClassifier()
DTModel.fit(X, Y)

# 预测新的数据点
Xnew = np.array([[2.5, 3.5], [4.5, 5.5]])
Ypred = DTModel.predict(Xnew)

# 输出新数据点的预测类别
print('新数据点的预测类别:')
for i, pred in enumerate(Ypred):
    print(f'Xnew[{i},:] -> 类别 {int(pred)}')

# 假设的真实类别，用于示例（实际应用中应有真实测试数据的标签）
Ytest = np.array([1, 2]) 

# 计算和显示混淆矩阵
confMat = confusion_matrix(Ytest, Ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=confMat)
disp.plot(cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.show()

# 可视化数据
plt.figure()
# 可视化训练数据
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], c='red', marker='o', label='类别 1')
plt.scatter(X[Y == 2][:, 0], X[Y == 2][:, 1], c='blue', marker='s', label='类别 2')

# 可视化新数据点
plt.scatter(Xnew[:, 0], Xnew[:, 1], c='black', marker='x', s=100, label='新数据点')

plt.title('决策树分类')
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.legend(loc='best')
plt.grid(True)
plt.show()
