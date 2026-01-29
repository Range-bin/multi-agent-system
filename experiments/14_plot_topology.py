# experiments/12_plot_by_topology.py
#生成“按拓扑分组”的箱线图
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_csv('results/consensus_precision_results.csv')

# 过滤有效数据
df_valid = df[df['Final_Std'] > 0]

# 按拓扑分组
topo_groups = df_valid.groupby('Topology')['Final_Std'].apply(list)
labels = list(topo_groups.index)
data = list(topo_groups.values)

# 绘图
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
plt.yscale('log')
plt.ylabel('最终状态标准差（对数尺度）')
plt.title('不同网络拓扑下的共识精度分布（所有策略与规模）')
plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()

# 保存
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/consensus_precision_by_topology.png', dpi=300, bbox_inches='tight')
print("✅ 拓扑分组图已保存: figures/consensus_precision_by_topology.png")

# 可选：显示
# plt.show()