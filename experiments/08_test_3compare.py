# experiments/05_strategy_comparison.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

print("=" * 60)
print("三种共识策略综合对比")
print("=" * 60)

strategies = [
    ('deGroot', 'DeGroot基准', {}, 'blue'),
    ('stubborn', '固执型 α=0.7', {'alpha': 0.7}, 'red'),
    ('susceptible', '从众型 β=2.0（等权重）', {'beta': 2.0}, 'green'),  # 改为β=2.0
]

results = []

for strategy_type, label, params, color in strategies:
    print(f"\n>>> 测试: {label}")
    
    sim = ConsensusSimulator(
        n_agents=5,
        topology='star',  # 星型网络效果明显
        initial_state_range=(0, 100),
        strategy=strategy_type,
        strategy_params=params
    )
    
    iterations = sim.run_until_convergence(max_iterations=100, tolerance=1e-4)
    history = sim.get_state_history()
    
    results.append({
        'label': label,
        'color': color,
        'iterations': iterations,
        'history': history,
        'consensus': np.mean(history[-1]),
        'initial_avg': np.mean(history[0]),
        'bias': np.mean(history[-1]) - np.mean(history[0]),
    })
    
    print(f"  收敛轮数: {iterations}")
    print(f"  共识值: {np.mean(history[-1]):.4f}")
    print(f"  初始平均: {np.mean(history[0]):.4f}")
    print(f"  偏差: {np.mean(history[-1]) - np.mean(history[0]):.4f}")

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 状态演化对比
ax1 = axes[0, 0]
for i, result in enumerate(results):
    history = result['history']
    for agent_id in range(5):
        ax1.plot(history[:, agent_id], color=result['color'], alpha=0.3)
    # 绘制平均线
    mean_line = np.mean(history, axis=1)
    ax1.plot(mean_line, color=result['color'], linewidth=2, label=result['label'])

ax1.set_xlabel('迭代轮数')
ax1.set_ylabel('状态值')
ax1.set_title('三种策略状态演化对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 收敛速度对比
ax2 = axes[0, 1]
labels = [r['label'] for r in results]
iterations = [r['iterations'] for r in results]
colors = [r['color'] for r in results]

bars = ax2.bar(labels, iterations, color=colors, alpha=0.7)
ax2.set_xlabel('策略类型')
ax2.set_ylabel('收敛所需轮数')
ax2.set_title('收敛速度对比')
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, iteration in zip(bars, iterations):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{iteration}', ha='center', va='bottom')

# 3. 共识值偏差对比
ax3 = axes[1, 0]
biases = [abs(r['bias']) for r in results]  # 取绝对值
bars2 = ax3.bar(labels, biases, color=colors, alpha=0.7)
ax3.set_xlabel('策略类型')
ax3.set_ylabel('共识值偏差绝对值')
ax3.set_title('共识值偏差对比')
ax3.grid(True, alpha=0.3, axis='y')

for bar, bias in zip(bars2, biases):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bias:.2f}', ha='center', va='bottom')

# 4. 收敛过程标准差对比
ax4 = axes[1, 1]
for result in results:
    history = result['history']
    std_history = [np.std(history[i]) for i in range(len(history))]
    ax4.plot(std_history, color=result['color'], linewidth=2, label=result['label'])

ax4.set_xlabel('迭代轮数')
ax4.set_ylabel('标准差')
ax4.set_title('收敛过程标准差变化')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')  # 对数坐标更清晰

plt.tight_layout()
plt.savefig('figures/strategy_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存至: figures/strategy_comparison.png")
plt.show()

print("\n" + "=" * 60)
print("实验结论总结:")
print("=" * 60)
for result in results:
    print(f"{result['label']}:")
    print(f"  • 收敛轮数: {result['iterations']}")
    print(f"  • 共识值偏差: {result['bias']:.4f}")
    print(f"  • 相对偏差: {abs(result['bias']/result['initial_avg'])*100:.2f}%")