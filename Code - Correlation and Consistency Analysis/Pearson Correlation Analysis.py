import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'simHei' # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 示例数据集
np.random.seed(0)  # 为了结果可重复
df = pd.DataFrame(np.random.randn(10, 10), columns=["M", "V", "D", "t", "w", "n", "fy", "fc", "L", "d"])
print(df.head())

# 计算相关性矩阵和p值矩阵
corr = df.corr()
n = len(df.columns)

# 初始化p值矩阵
pval = pd.DataFrame(np.ones((n, n)), columns=df.columns, index=df.columns)

# 计算每对变量的p值
for i in range(n):
    for j in range(n):
        if i == j:
            pval.iloc[i, j] = np.nan  # 对角线没有p值
        else:
            corr_coef, p = stats.pearsonr(df.iloc[:, i], df.iloc[:, j])
            pval.iloc[i, j] = p

# 定义函数将p值转换为星号
def pval_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2.5 * n))

# 绘制每个位置的直方图、散点图及相关性热图
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == j:
            # 对角线：绘制直方图
            sns.histplot(df.iloc[:, i], kde=True, ax=ax, color='skyblue')
        elif i > j:
            # 下三角：绘制带线性拟合的散点图
            sns.regplot(x=df.iloc[:, j], y=df.iloc[:, i], ax=ax,
                        scatter_kws={'s': 20, 'alpha': 0.7}, line_kws={'color': 'red'})
        else:
            # 上三角：绘制热图显示相关系数，并添加显著性星号
            corr_val = corr.iloc[i, j]
            p_val = pval.iloc[i, j]
            stars = pval_to_stars(p_val)
            
            # 绘制热图
            sns.heatmap(pd.DataFrame([[corr_val]]), cmap=sns.diverging_palette(240, 10, as_cmap=True),
                        cbar=False, annot=False, fmt=".2f", square=True, ax=ax, vmin=-1, vmax=1)
            
            # 在热图上方添加星号
            if stars:
                ax.text(0.5, 0.7, stars, ha='center', va='center', color='black', fontsize=14, fontweight='bold')
            
            # 在热图下方添加相关系数
            ax.text(0.5, 0.3, f"{corr_val:.2f}", ha='center', va='center', color='black', fontsize=12)
        
        # 隐藏不需要的轴标签
        if i < n - 1:
            ax.set_xticklabels([])
        if j > 0:
            ax.set_yticklabels([])

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# 在图形旁边添加全局色条
fig.subplots_adjust(right=0.85)  # 调整图形右侧空间以显示色条
cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # 定义色条位置和大小
norm = plt.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap=sns.diverging_palette(240, 10, as_cmap=True), norm=norm)
sm.set_array([])  # 为空数组设置色条
fig.colorbar(sm, cax=cbar_ax)  # 添加全局色条

# 添加显著性星号说明
legend_text = '显著性水平:\n*** p < 0.001\n** p < 0.01\n* p < 0.05'
fig.text(0.9, 0.05, legend_text, fontsize=12, va='bottom', ha='left')

plt.show()