# experiments/09_topology_comparison.py 验证：只有在不对称拓扑（如星型）中，共识值才会偏离全局平均；在对称拓扑（如环形）中，共识值 ≈ 全局平均。
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
print("补充实验：不同拓扑下三种策略的共识值对比 (N=10)")
print("=" * 70)

# 固定初始状态（确保可比性）
np.random.seed(42)
initial_states = np.random.uniform(0, 100, 10)
global_avg = np.mean(initial_states)
print(f"全局初始平均: {global_avg:.4f}")

# 测试拓扑
topologies = ['ring', 'star', 'complete']
topology_names = {'ring': '环形网络', 'star': '星型网络', 'complete': '全连接网络'}

# 测试策略
strategies = [
    ('deGroot', 'DeGroot', {}),
    ('stubborn', '固执型 α=0.7', {'alpha': 0.7}),
    ('susceptible', '易受影响 β=2.0', {'beta': 2.0})
]

results = {}

for topo in topologies:
    print(f"\n>>> 拓扑: {topology_names[topo]}")
    results[topo] = {}
    
    for strat_type, label, params in strategies:
        # 创建模拟器（传入固定初始状态）
        sim = ConsensusSimulator(
            n_agents=10,
            topology=topo,
            initial_state_range=(0, 100),
            strategy=strat_type,
            strategy_params=params
        )
        # 强制使用相同初始状态
        sim.state_history[0] = initial_states.copy()
        for i in range(10):
            sim.agents[i].state = initial_states[i]
        
        iterations = sim.run_until_convergence(max_iterations=200, tolerance=1e-5, verbose=False)
        history = sim.get_state_history()
        consensus_val = np.mean(history[-1])
        bias = consensus_val - global_avg
        
        results[topo][label] = {
            'consensus': consensus_val,
            'bias': bias,
            'iterations': iterations
        }
        
        print(f"  {label}: 共识={consensus_val:.4f}, 偏向={bias:+.4f}, 轮数={iterations}")

# 绘制共识值对比图
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(strategies))
width = 0.25

for i, (topo, name) in enumerate([('ring', '环形'), ('star', '星型'), ('complete', '全连接')]):
    consensus_vals = [results[topo][label]['consensus'] for _, label, _ in strategies]
    bars = ax.bar(x + i*width, consensus_vals, width, label=name)

ax.axhline(global_avg, color='red', linestyle='--', label=f'全局平均 ({global_avg:.2f})')
ax.set_xlabel('策略类型')
ax.set_ylabel('共识值')
ax.set_title('不同拓扑下共识值对比 (N=10)')
ax.set_xticks(x + width)
ax.set_xticklabels([label for _, label, _ in strategies])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/topology_consensus_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存: figures/topology_consensus_comparison.png")