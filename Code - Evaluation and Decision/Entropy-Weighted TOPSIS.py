import numpy as np
import pandas as pd

# 输入数据，列表代表每个评价因子的值
X1 = [200, 150, 180]  # 正面指标
X2 = [300, 350, 330]  # 正面指标
X3 = [400, 250, 420]  # 正面指标

Y1 = [50, 80, 45]     # 负面指标
Y2 = [70, 40, 65]     # 负面指标
Y3 = [60, 50, 55]     # 负面指标

# 各个方案的名字
names = ["方案A", "方案B", "方案C"]

# 将列表合并为正面和负面指标矩阵
positive_indices = np.array([X1, X2, X3]).T
negative_indices = np.array([Y1, Y2, Y3]).T

# 数据标准化
# 正面指标标准化
norm_pos = positive_indices / np.sum(positive_indices, axis=0)

# 负面指标标准化
norm_neg = np.min(negative_indices, axis=0) / negative_indices

# 合并标准化后的矩阵
decision_matrix = np.hstack((norm_pos, norm_neg))
m, n = decision_matrix.shape

# 计算熵值和权重
entropy = np.zeros(n)
for j in range(n):
    p = decision_matrix[:, j] / np.sum(decision_matrix[:, j])
    entropy[j] = -np.sum(p * np.log(p + np.finfo(float).eps))  # eps防止对0取对数

entropy = entropy / np.log(m)
weight = (1 - entropy) / np.sum(1 - entropy)

# 构建加权标准化决策矩阵
weighted_matrix = decision_matrix * weight

# 确定理想解和负理想解
ideal_solution = np.max(weighted_matrix, axis=0)
negative_ideal_solution = np.min(weighted_matrix, axis=0)

# 计算距离
distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
distance_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))

# 计算相对接近度
relative_closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

# 排序
sorted_indices = np.argsort(relative_closeness)[::-1]
sorted_closeness = relative_closeness[sorted_indices]
sorted_names = [names[i] for i in sorted_indices]

# 输出结果
rankings = pd.DataFrame({
    "排名": range(1, len(sorted_names) + 1),
    "方案名称": sorted_names,
    "相对接近度": sorted_closeness
})

print(rankings)
