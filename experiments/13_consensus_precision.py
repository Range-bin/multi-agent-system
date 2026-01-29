# experiments/11_consensus_precision.py
"""
共识精度分析实验
目标：验证不同策略、拓扑和规模下，系统是否达成高精度共识（最终状态标准差 < 1e-5）
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

print("=" * 70)
print("共识精度分析实验：验证最终一致性水平")
print("=" * 70)

# 固定随机种子以确保可复现性
np.random.seed(42)

# 实验配置
topologies = ['ring', 'star', 'complete']
strategies = [
    ('deGroot', 'DeGroot', {}),
    ('stubborn', '固执型 α=0.7', {'alpha': 0.7}),
    ('susceptible', '易受影响 β=2.0', {'beta': 2.0})
]
sizes = [10, 20, 50]  # 不测试100以节省时间，但逻辑支持

# 存储结果
results = []

for n in sizes:
    print(f"\n>>> 智能体数量 N = {n}")
    # 生成固定初始状态（保证跨拓扑可比）
    initial_states = np.random.uniform(0, 100, n)
    
    for topo in topologies:
        for strat_type, label, params in strategies:
            try:
                sim = ConsensusSimulator(
                    n_agents=n,
                    topology=topo,
                    initial_state_range=(0, 100),
                    strategy=strat_type,
                    strategy_params=params
                )
                # 强制使用相同初始状态
                for i in range(n):
                    sim.agents[i].state = initial_states[i]
                sim.state_history[0] = initial_states.copy()
                
                iterations = sim.run_until_convergence(
                    max_iterations=2000,
                    tolerance=1e-8,
                    verbose=(n == 50 and topo == 'ring')  # 只对 N=50 ring 打印详细日志
                )
                final_states = sim.get_state_history()[-1]
                final_std = np.std(final_states)
                consensus_val = np.mean(final_states)
                
                results.append({
                    'N': n,
                    'Topology': topo,
                    'Strategy': label,
                    'Iterations': iterations,
                    'Final_Std': final_std,
                    'Consensus_Value': consensus_val
                })
                
                print(f"  {topo:8} | {label:15} → 轮数={iterations:3d}, 最终标准差={final_std:.2e}")
                
            except Exception as e:
                print(f"  ❌ {topo} | {label} → 出错: {e}")
                results.append({
                    'N': n,
                    'Topology': topo,
                    'Strategy': label,
                    'Iterations': -1,
                    'Final_Std': np.nan,
                    'Consensus_Value': np.nan
                })

# 保存为CSV
df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/consensus_precision_results.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ 结果已保存至: results/consensus_precision_results.csv")

# 可视化：最终标准差分布（箱线图）
plt.figure(figsize=(12, 6))
df_valid = df[df['Final_Std'] > 0]

# 按策略分组绘制箱线图
strategies_clean = [r['Strategy'] for r in results if not np.isnan(r['Final_Std'])]
std_values = [r['Final_Std'] for r in results if not np.isnan(r['Final_Std'])]

# 构建绘图数据
strategy_groups = {}
for _, row in df_valid.iterrows():
    key = row['Strategy']
    if key not in strategy_groups:
        strategy_groups[key] = []
    strategy_groups[key].append(row['Final_Std'])

# 绘图
labels = list(strategy_groups.keys())
data = [strategy_groups[k] for k in labels]

plt.boxplot(data, labels=labels, patch_artist=True)
plt.yscale('log')
plt.ylabel('最终状态标准差（对数尺度）')
plt.title('不同策略下的共识精度分布（所有拓扑与规模）')
plt.grid(True, alpha=0.3, which='both')
plt.xticks(rotation=15)

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/consensus_precision_boxplot.png', dpi=300, bbox_inches='tight')
print("✅ 精度分布图已保存: figures/consensus_precision_boxplot.png")

# 打印统计摘要
print("\n📊 精度统计摘要:")
print(df_valid[['Strategy', 'Final_Std']].groupby('Strategy')['Final_Std'].describe())

# 判断是否全部满足精度要求
all_good = (df_valid['Final_Std'] < 1e-5).all()
if all_good:
    print("\n✅ 所有实验均满足高精度共识要求（标准差 < 1e-5）")
else:
    print("\n⚠️ 部分实验未达到预期精度，请检查异常项")