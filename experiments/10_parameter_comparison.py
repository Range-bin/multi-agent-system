# experiments/06_parameter_comparison.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

print("=" * 70)
print("两种策略参数对比分析")
print("=" * 70)

# 测试固执型不同α值
print("\n>>> 固执型策略参数分析")
alpha_values = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
alpha_results = []

for alpha in alpha_values:
    sim = ConsensusSimulator(
        n_agents=5,
        topology='star',
        initial_state_range=(0, 100),
        strategy='stubborn',
        strategy_params={'alpha': alpha}
    )
    iterations = sim.run_until_convergence(max_iterations=200, tolerance=1e-4, verbose=False)
    history = sim.get_state_history()
    
    alpha_results.append({
        'alpha': alpha,
        'iterations': iterations,
        'consensus': np.mean(history[-1]),
        'initial_avg': np.mean(history[0]),
    })

# 测试易受影响型不同β值
print("\n>>> 易受影响型策略参数分析")
beta_values = [1.0, 1.5, 2.0, 3.0, 5.0]
beta_results = []

for beta in beta_values:
    sim = ConsensusSimulator(
        n_agents=5,
        topology='star',
        initial_state_range=(0, 100),
        strategy='susceptible',
        strategy_params={'beta': beta}
    )
    iterations = sim.run_until_convergence(max_iterations=200, tolerance=1e-4, verbose=False)
    history = sim.get_state_history()
    
    beta_results.append({
        'beta': beta,
        'iterations': iterations,
        'consensus': np.mean(history[-1]),
        'initial_avg': np.mean(history[0]),
    })

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 参数-收敛速度对比
ax1 = axes[0, 0]
# 固执型
alpha_vals = [r['alpha'] for r in alpha_results]
alpha_iters = [r['iterations'] for r in alpha_results]
ax1.plot(alpha_vals, alpha_iters, 'ro-', linewidth=2, markersize=8, label='固执型 (α)')

# 易受影响型（转换为等效固执度：1/β）
beta_vals = [1.0/r['beta'] if r['beta'] > 0 else 0 for r in beta_results]  # 等效固执度
beta_iters = [r['iterations'] for r in beta_results]
ax1.plot(beta_vals, beta_iters, 'bo-', linewidth=2, markersize=8, label='易受影响型 (1/β)')

ax1.set_xlabel('固执程度 (α 或 1/β)')
ax1.set_ylabel('收敛所需轮数')
ax1.set_title('两种策略参数对收敛速度的影响')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 参数-共识值对比
ax2 = axes[0, 1]
alpha_biases = [abs(r['consensus'] - r['initial_avg']) for r in alpha_results]
beta_biases = [abs(r['consensus'] - r['initial_avg']) for r in beta_results]

ax2.plot(alpha_vals, alpha_biases, 'ro-', linewidth=2, markersize=8, label='固执型')
ax2.plot(beta_vals, beta_biases, 'bo-', linewidth=2, markersize=8, label='易受影响型')

ax2.set_xlabel('固执程度 (α 或 1/β)')
ax2.set_ylabel('共识值偏差')
ax2.set_title('两种策略参数对共识值的影响')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 收敛速度分布对比
ax3 = axes[1, 0]
strategies_data = [
    ('固执型 α=0.7', alpha_results[3]['iterations'], 'red'),
    ('从众型 β=2.0', beta_results[2]['iterations'], 'blue'),
    ('DeGroot基准', alpha_results[0]['iterations'], 'green'),
]

labels = [d[0] for d in strategies_data]
values = [d[1] for d in strategies_data]
colors = [d[2] for d in strategies_data]

bars = ax3.bar(labels, values, color=colors, alpha=0.7)
ax3.set_ylabel('收敛轮数')
ax3.set_title('三种典型策略收敛速度对比')
ax3.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{value}', ha='center', va='bottom')

# 4. 参数权衡关系
ax4 = axes[1, 1]
# 固执型数据点
for r in alpha_results:
    ax4.scatter(r['iterations'], abs(r['consensus'] - r['initial_avg']), 
               c='red', s=100, alpha=0.6, label='固执型' if r['alpha']==0.7 else '')

# 易受影响型数据点
for r in beta_results:
    ax4.scatter(r['iterations'], abs(r['consensus'] - r['initial_avg']),
               c='blue', s=100, alpha=0.6, label='易受影响型' if r['beta']==2.0 else '')

ax4.set_xlabel('收敛轮数')
ax4.set_ylabel('共识值偏差')
ax4.set_title('收敛速度与共识质量的权衡关系')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/parameter_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n对比图表已保存至: figures/parameter_comparison.png")

# 打印关键数据
print("\n" + "=" * 70)
print("关键性能数据")
print("=" * 70)
print("\n固执型策略 (α变化):")
print(f"{'α':<6} {'收敛轮数':<10} {'共识值':<10} {'偏差':<10}")
for r in alpha_results:
    print(f"{r['alpha']:<6} {r['iterations']:<10} {r['consensus']:<10.4f} "
          f"{r['consensus']-r['initial_avg']:<10.4f}")

print("\n易受影响型策略 (β变化):")
print(f"{'β':<6} {'收敛轮数':<10} {'共识值':<10} {'偏差':<10}")
for r in beta_results:
    print(f"{r['beta']:<6} {r['iterations']:<10} {r['consensus']:<10.4f} "
          f"{r['consensus']-r['initial_avg']:<10.4f}")