import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# 设置中文字体和负号正常显示
plt.rcParams['font.family'] = 'SimHei'  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义特征列表
X1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X2 = [1, 2, 2, 6, 6, 7, 8, 9, 1, 3]
# 可以添加更多特征列表，如X3, X4...

# 将特征列表合并为数据集X
X = np.column_stack((X1, X2))  # 合并为二维数据集
# 如果有更多特征，可以在np.column_stack中加入更多列表

# 定义评估指标函数
def evaluate_clustering(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    return silhouette_avg, ch_score, db_score

# 标准化函数
def standardize_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    standardized_scores = (scores - mean) / std
    return standardized_scores

# 计算簇内点到簇中心的距离
def compute_distances_to_centers(X, labels, centers):
    distances = []
    for i in range(len(centers)):
        cluster_points = X[labels == i]
        center = centers[i]
        distances.extend(np.linalg.norm(cluster_points - center, axis=1))
    return np.array(distances)

# 判断数据是否是球状
def is_data_spherical(distances, threshold=0.95):
    # 估计球的半径（使用距离的百分位数）
    estimated_radius = np.percentile(distances, threshold * 100)
    # 计算在估计的球内的点的比例
    in_sphere_ratio = np.mean(distances <= estimated_radius)
    return in_sphere_ratio

# 网格搜索K-means++的最优聚类数量
def find_best_k(X, k_range):
    best_k = 0
    best_score = -float('inf')
    best_kmeans_labels = None
    best_kmeans_centers = None
    all_silhouette_scores = []
    all_ch_scores = []
    all_db_scores = []

    for k in k_range:
        if k <= 1 or k >= X.shape[0]:
            continue

        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)  # 设置 init='k-means++'
        kmeans_labels = kmeans.fit_predict(X)
        silhouette_avg, ch_score, db_score = evaluate_clustering(X, kmeans_labels)
        all_silhouette_scores.append(silhouette_avg)
        all_ch_scores.append(ch_score)
        all_db_scores.append(db_score)

    # 标准化所有的评分
    standardized_silhouette_scores = standardize_scores(np.array(all_silhouette_scores))
    standardized_ch_scores = standardize_scores(np.array(all_ch_scores))
    standardized_db_scores = standardize_scores(np.array(all_db_scores))

    for i, k in enumerate(k_range):
        if k <= 1 or k >= X.shape[0]:
            continue
        combined_score = (standardized_silhouette_scores[i] + standardized_ch_scores[i] - standardized_db_scores[i])

        if combined_score > best_score:
            best_k = k
            best_score = combined_score
            best_kmeans_labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
            best_kmeans_centers = KMeans(n_clusters=k, random_state=42).fit(X).cluster_centers_

    return best_k, best_kmeans_labels, best_kmeans_centers, (standardized_silhouette_scores, standardized_ch_scores, standardized_db_scores)
# 设定k值的范围
k_range = range(2, 11)  # 确保最小k值为2，避免轮廓系数计算错误

# 找到最佳k值
best_k, best_kmeans_labels, best_kmeans_centers, best_scores = find_best_k(X, k_range)

print(f"最佳k值: {best_k}")
print("此时的标准化轮廓系数，CH和DB指数:", best_scores)
print(f"最佳综合得分: {best_scores[0][best_k - 2] + best_scores[1][best_k - 2] - best_scores[2][best_k - 2]}")

# 计算簇内点到簇中心的距离
distances = compute_distances_to_centers(X, best_kmeans_labels, best_kmeans_centers)

# 估计球的半径（使用距离的百分位数）
estimated_radius = np.percentile(distances, 95)  # 使用95%的分位数作为球的半径
print(f"估计的球形簇的半径：{estimated_radius}")

# 计算在估计的球内的点的比例
def is_data_spherical(distances, threshold=0.95):
    estimated_radius = np.percentile(distances, threshold * 100)
    in_sphere_ratio = np.mean(distances <= estimated_radius)
    return in_sphere_ratio

in_sphere_ratio = is_data_spherical(distances)
print(f"数据点在估计的球内的比例: {in_sphere_ratio:.2f}")
if in_sphere_ratio > 0.9:  # 阈值可调，通常选择0.8到0.9之间
    print("数据呈球状分布")
else:
    print("数据不是球状分布")


# 可视化聚类结果
plt.figure()
for i in range(best_k):
    plt.scatter(X[best_kmeans_labels == i, 0], X[best_kmeans_labels == i, 1], label=f'聚类{i+1}')
plt.scatter(best_kmeans_centers[:, 0], best_kmeans_centers[:, 1], marker='x', color='k', s=100, linewidths=2, label='聚类中心')
plt.title(f'最佳k值: {best_k} 的 K-means 聚类结果')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='best')
plt.show()