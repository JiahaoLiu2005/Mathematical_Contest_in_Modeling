import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成一些分类数据
np.random.seed(0)  # 保证结果可重复
X = np.vstack((np.random.rand(10, 2) * 5, np.random.rand(10, 2) * 5 + 1))
Y = np.hstack((np.ones(10), np.ones(10) * 2))

# 拆分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 使用支持向量机进行分类
SVMModel = svm.SVC(kernel='rbf', C=1, gamma='scale', probability=True)
SVMModel.fit(X_train, Y_train)

# 预测测试集数据
Y_pred = SVMModel.predict(X_test)

# 输出评价指标
report = classification_report(Y_test, Y_pred, target_names=['类别 1', '类别 2'], zero_division=0, output_dict=True)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

print(f"模型总体精确率: {precision:.2f}")
print(f"模型总体召回率: {recall:.2f}")
print(f"模型总体F1评分: {f1_score:.2f}")

# 新数据点的预测
Xnew = np.array([[2.5, 3.5], [4.5, 5.5]])
Ynew_pred = SVMModel.predict(Xnew)
print('新数据点的预测类别:')
for i, pred in enumerate(Ynew_pred):
    print(f"Xnew[{i},:] -> 类别 {int(pred)}")

# 绘制ROC曲线
Y_test_bin = label_binarize(Y_test, classes=[1, 2])
Y_pred_prob = SVMModel.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(Y_test_bin, Y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC 曲线 (面积 = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真正例率')
plt.title('接收者操作特征（ROC）曲线')
plt.legend(loc='lower right')
plt.show()

