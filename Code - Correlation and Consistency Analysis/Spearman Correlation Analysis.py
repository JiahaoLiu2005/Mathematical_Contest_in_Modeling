import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.cm as cm

sns.set(font='SimHei')
plt.rcParams['axes.unicode_minus'] = False

def rank_data(data):
    sorted_indices = sorted(range(len(data)), key=lambda k: data[k])
    ranks = [0] * len(data)
    for i, rank in enumerate(sorted_indices):
        ranks[rank] = i + 1
    return ranks

def spearman_corr(X, Y):
    X_rank = rank_data(X)
    Y_rank = rank_data(Y)
    n = len(X)
    diff = np.array(X_rank) - np.array(Y_rank)
    return 1 - (6 * np.sum(diff ** 2)) / (n * (n ** 2 - 1))

def plot_graphs(variables, data):
    correlations = {}
    corr_matrix = np.zeros((len(variables), len(variables)))
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i != j:
                corr = spearman_corr(data[i], data[j])
                correlations[(variables[i], variables[j])] = corr
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = 1  # 自身的相关系数为1

    main_var = variables[0]
    main_var_corr = {var: spearman_corr(data[0], data[i]) for i, var in enumerate(variables)}

    # 创建图形布局
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 绘制相关性网络图
    G = nx.Graph()
    for var in variables:
        G.add_node(var, size=abs(main_var_corr[var]) * 1000, color=cm.coolwarm((main_var_corr[var] + 1) / 2))
    for (var1, var2), corr in correlations.items():
        G.add_edge(var1, var2, weight=abs(corr))

    angle_step = 2 * np.pi / len(variables)
    pos = {var: (np.cos(i * angle_step), np.sin(i * angle_step)) for i, var in enumerate(variables)}

    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_size=[G.nodes[var]['size'] for var in G.nodes],
                           node_color=[G.nodes[var]['color'] for var in G.nodes], alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=axes[0], width=[G[u][v]['weight'] * 2 for u, v in G.edges], alpha=0.6)
    nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=12, font_family='SimHei')

    edge_labels = {(u, v): f"{correlations[(u, v)]:+.2f}" for u, v in G.edges()}
    label_offset = 0.1
    for (u, v), label in edge_labels.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        x = (x1 + x2) / 2 + np.sign(x1 - x2) * label_offset
        y = (y1 + y2) / 2 + np.sign(y1 - y2) * label_offset
        axes[0].text(x, y, label, size=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    axes[0].set_title('变量之间的Spearman相关性网络图', fontproperties="SimHei")
    axes[0].axis('off')

    # 绘制热力图
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', xticklabels=variables, yticklabels=variables, ax=axes[1])
    axes[1].set_title('Spearman相关性热力图', fontproperties="SimHei")

    plt.tight_layout()
    plt.show()

    # 打印三线表
    print("变量\tSpearman相关系数")
    for i in range(1, len(variables)):
        print(f"{variables[i]}\t{main_var_corr[variables[i]]:.2f}")

if __name__ == '__main__':
    # 示例数据
    variables = ["变量A", "变量B", "变量C", "变量D"]
    data = [
        [70, 70, 72, 73, 70, 69, 68, 70, 73],
        [1630, 1740, 1840, 1930, 2010, 2040, 2070, 2080, 2050],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    plot_graphs(variables, data)
