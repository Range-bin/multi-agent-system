# experiments/04_test_susceptible.py
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
print("新版易受影响型策略全面测试 (β≥1.0)")
print("=" * 70)

# 新的β值列表：覆盖从轻度固执到高度从众
beta_values = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
results = []

for beta in beta_values:
    print(f"\n{'='*40}")
    print(f">>> 测试 β = {beta}")
    print(f"{'='*40}")
    
    # 显示权重信息
    if beta == 1.0:
        print("  权重: DeGroot行为 (自身与邻居等权平均)")
    else:
        self_weight = 1.0 / beta
        neighbor_weight = (beta - 1.0) / beta
        print(f"  权重: 自身={self_weight:.3f}, 邻居={neighbor_weight:.3f}")
        print(f"  行为: {'轻度固执' if beta < 2.0 else '等权重' if beta == 2.0 else '从众'}")
    
    # 创建模拟器
    sim = ConsensusSimulator(
        n_agents=5,
        topology='star',  # 星型网络效果明显
        initial_state_range=(0, 100),
        strategy='susceptible',
        strategy_params={'beta': beta}
    )
    
    # 运行仿真
    iterations = sim.run_until_convergence(max_iterations=200, tolerance=1e-4)
    history = sim.get_state_history()
    
    # 收集结果
    consensus = np.mean(history[-1])
    initial_avg = np.mean(history[0])
    bias = consensus - initial_avg
    final_std = np.std(history[-1])
    
    results.append({
        'beta': beta,
        'iterations': iterations,
        'consensus': consensus,
        'initial_avg': initial_avg,
        'bias': bias,
        'final_std': final_std,
        'self_weight': 1.0/beta if beta > 1.0 else 0.5,  # β=1.0时记为0.5
        'neighbor_weight': (beta-1.0)/beta if beta > 1.0 else 0.5,
    })
    
    print(f"  收敛轮数: {iterations}")
    print(f"  共识值: {consensus:.4f}")
    print(f"  初始平均: {initial_avg:.4f}")
    print(f"  偏差: {bias:.4f}")
    print(f"  相对偏差: {abs(bias/initial_avg)*100:.2f}%")
    print(f"  最终标准差: {final_std:.6f}")

# 绘制结果分析图
print(f"\n{'='*70}")
print("易受影响型策略性能分析")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. β值 vs 收敛轮数
ax1 = axes[0, 0]
betas = [r['beta'] for r in results]
iterations = [r['iterations'] for r in results]
ax1.plot(betas, iterations, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('β值')
ax1.set_ylabel('收敛所需轮数')
ax1.set_title('β值对收敛速度的影响')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')  # 对数坐标显示更清晰

# 标记特殊点
special_points = [(1.0, 'DeGroot'), (2.0, '等权重')]
for beta, label in special_points:
    idx = betas.index(beta)
    ax1.annotate(label, xy=(beta, iterations[idx]), 
                xytext=(beta*1.2, iterations[idx]*1.1),
                arrowprops=dict(arrowstyle='->', alpha=0.7))

# 2. β值 vs 共识值偏差
ax2 = axes[0, 1]
biases = [abs(r['bias']) for r in results]  # 取绝对值
ax2.plot(betas, biases, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('β值')
ax2.set_ylabel('共识值偏差绝对值')
ax2.set_title('β值对共识值偏差的影响')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

# 3. 自身权重 vs 收敛速度
ax3 = axes[1, 0]
self_weights = [r['self_weight'] for r in results]
ax3.plot(self_weights, iterations, 'go-', linewidth=2, markersize=8)
ax3.set_xlabel('自身权重')
ax3.set_ylabel('收敛所需轮数')
ax3.set_title('自身权重对收敛速度的影响')
ax3.grid(True, alpha=0.3)

# 4. 收敛速度与偏差的关系
ax4 = axes[1, 1]
scatter = ax4.scatter(iterations, biases, c=betas, cmap='viridis', s=100, alpha=0.7)
ax4.set_xlabel('收敛轮数')
ax4.set_ylabel('共识值偏差')
ax4.set_title('收敛速度与偏差的权衡关系')
ax4.grid(True, alpha=0.3)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('β值')

plt.tight_layout()
plt.savefig('figures/new_susceptible_analysis.png', dpi=300, bbox_inches='tight')
print(f"分析图表已保存至: figures/new_susceptible_analysis.png")

# 打印汇总表
print(f"\n{'='*70}")
print("易受影响型策略性能汇总表")
print("=" * 70)
print(f"{'β值':<6} {'自身权重':<8} {'邻居权重':<8} {'收敛轮数':<10} {'共识值':<10} {'偏差':<10} {'相对偏差':<12}")
print("-" * 70)
for r in results:
    print(f"{r['beta']:<6} {r['self_weight']:<8.3f} {r['neighbor_weight']:<8.3f} "
          f"{r['iterations']:<10} {r['consensus']:<10.4f} {r['bias']:<10.4f} "
          f"{abs(r['bias']/r['initial_avg'])*100:<10.2f}%")

print("\n📊 关键结论:")
print("1. β越小（越接近1）→ 收敛越慢（越固执）")
print("2. β越大（越远离1）→ 收敛越快（越从众）")
print("3. β=2.0时自身与邻居等权重")
print("4. β=1.0时退化为DeGroot基准")