# experiments/10_scalability_test.py 规模可扩展性测试 验证随着 N 增大，收敛时间如何变化？星型网络是否存在瓶颈？
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
print("补充实验：规模可扩展性测试 (星型网络, Susceptible β=2.0)")
print("=" * 70)

sizes = [10, 20, 50, 100]
strategy = 'susceptible'
params = {'beta': 2.0}
topology = 'star'

results = []

for n in sizes:
    print(f"\n>>> N = {n}")
    # 固定随机种子，但允许不同规模有不同初始值
    np.random.seed(42)
    sim = ConsensusSimulator(
        n_agents=n,
        topology=topology,
        initial_state_range=(0, 100),
        strategy=strategy,
        strategy_params=params
    )
    
    iterations = sim.run_until_convergence(max_iterations=500, tolerance=1e-5, verbose=False)
    final_std = np.std(sim.get_state_history()[-1])
    
    results.append({
        'n': n,
        'iterations': iterations,
        'final_std': final_std
    })
    
    print(f"  收敛轮数: {iterations}, 最终标准差: {final_std:.6f}")

# 计算可扩展性系数 κ(n)
base_n = results[0]['n']
base_iter = results[0]['iterations']
kappas = []
for r in results:
    kappa = (r['iterations'] / base_iter) * (base_n / r['n'])
    kappas.append(kappa)
    print(f"N={r['n']}: κ = {kappa:.3f}")

# 绘图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 收敛轮数
ax1.plot(sizes, [r['iterations'] for r in results], 'bo-', label='收敛轮数')
ax1.set_xlabel('智能体数量 N')
ax1.set_ylabel('收敛轮数')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# 可扩展性系数（次坐标轴）
ax2 = ax1.twinx()
ax2.plot(sizes, kappas, 'ro--', label='可扩展性系数 κ(n)')
ax2.set_ylabel('κ(n)')
ax2.legend(loc='upper right')

plt.title('星型网络规模可扩展性测试 (β=2.0)')
plt.tight_layout()
plt.savefig('figures/scalability_test.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存: figures/scalability_test.png")