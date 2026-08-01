import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 设置字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据
cluster1 = np.random.normal(loc=[2, 2], scale=[0.5, 0.5], size=[20, 2])
cluster2 = np.random.normal(loc=[5, 5], scale=[0.5, 0.5], size=[20, 2])
data = np.vstack((cluster1, cluster2))

# 参数网格
eps_values = np.arange(0.5, 2.0, 0.1)
min_samples_values = range(2, 10)

# 存储最佳参数和对应的评估分数
best_params = None
best_silhouette_score = -1
best_noise_ratio = float('inf')
best_dbl_score = float('inf')
best_ch_score = -1

# 网格搜索
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data)
        
        # 计算噪声点比率
        noise_ratio = np.sum(clusters == -1) / len(clusters)
        
        # 去除噪声点
        clustered_points = data[clusters != -1]
        labels_clustered = clusters[clusters != -1]
        
        if len(np.unique(labels_clustered)) > 1:  # 确保有至少两个聚类
            silhouette_avg = silhouette_score(clustered_points, labels_clustered)
            dbl_score = davies_bouldin_score(clustered_points, labels_clustered)
            ch_score = calinski_harabasz_score(clustered_points, labels_clustered)
            
            # 更新最佳参数
            if silhouette_avg > best_silhouette_score and noise_ratio < best_noise_ratio and dbl_score < best_dbl_score and ch_score > best_ch_score:
                best_silhouette_score = silhouette_avg
                best_noise_ratio = noise_ratio
                best_dbl_score = dbl_score
                best_ch_score = ch_score
                best_params = (eps, min_samples)

# 输出最佳参数和评估分数
print(f"最佳参数: eps={best_params[0]}, min_samples={best_params[1]}")
print(f"最佳轮廓系数: {best_silhouette_score}")
print(f"噪声点比率: {best_noise_ratio}")
print(f"最佳DBL指数: {best_dbl_score}")
print(f"最佳CH指数: {best_ch_score}")

# 重新绘制图表，确保类别1和类别2分别用绿色和黄色标记，噪声点用黑色标记

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(data)

# 为了在图例中区分噪声，我们需要创建不同的颜色映射
# 创建一个颜色列表，其中噪声用黑色表示
colors = ['green' if label == 1 else 'yellow' if label == 0 else 'black' for label in clusters]

# 绘制聚类结果
plt.scatter(data[:, 0], data[:, 1], c=colors, marker='o', edgecolor='k')

# 添加图例
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='类别1', markerfacecolor='green', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='类别2', markerfacecolor='yellow', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='噪声', markerfacecolor='black', markersize=10)],
           loc='upper right')

plt.title('DBSCAN聚类结果')
plt.xlabel('特征 x')
plt.ylabel('特征 y')
plt.show()
