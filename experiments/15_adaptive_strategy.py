# experiments/15_adaptive_strategy.py （终极修正版）

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator
from src.strategies import AdaptiveSusceptibleStrategy, DiffAdaptiveStrategy  # ← 新增导入

def create_adaptive_simulator(n_agents, topology, initial_state_range, strategy, max_iterations):
    """通用创建函数，支持任意策略"""
    sim = ConsensusSimulator(
        n_agents=n_agents,
        topology=topology,
        initial_state_range=initial_state_range,
        strategy='deGroot',
        max_iterations=max_iterations,
        verbose=False
    )
    for agent in sim.agents.values():
        agent.strategy = strategy
    return sim

def main():
    print("🚀 启动自适应策略对比实验（终极修正版）...\n")
    
    N_AGENTS = 20
    TOPOLOGY = 'ring'
    INITIAL_RANGE = (0, 100)
    MAX_ITER = 500
    TOLERANCE = 1e-4

    # ===== 固定策略：beta=2.0（邻居权重=0.5）=====
    print("▶ 运行固定β策略 (SusceptibleStrategy, beta=2.0)...")
    sim_fixed = ConsensusSimulator(
        n_agents=N_AGENTS,
        topology=TOPOLOGY,
        initial_state_range=INITIAL_RANGE,
        strategy='susceptible',
        strategy_params={'beta': 2.0},
        max_iterations=MAX_ITER,
        verbose=False
    )
    steps_fixed = sim_fixed.run_until_convergence(tolerance=TOLERANCE, verbose=False)
    avg_fixed = [np.mean(states) for states in sim_fixed.get_state_history()]

    # ===== 新自适应策略：基于差异的自适应 =====
    print("▶ 运行新自适应策略 (DiffAdaptiveStrategy, beta_max=0.7, k=0.05)...")
    adaptive_strategy = DiffAdaptiveStrategy(beta_max=0.7, k=0.05)
    sim_adaptive = create_adaptive_simulator(
        n_agents=N_AGENTS,
        topology=TOPOLOGY,
        initial_state_range=INITIAL_RANGE,
        strategy=adaptive_strategy,
        max_iterations=MAX_ITER
    )
    steps_adaptive = sim_adaptive.run_until_convergence(tolerance=TOLERANCE, verbose=False)
    avg_adaptive = [np.mean(states) for states in sim_adaptive.get_state_history()]

    # ===== 结果输出 =====
    print("\n" + "="*50)
    print("📊 实验结果对比（新自适应策略）")
    print("="*50)
    print(f"固定β策略       : {steps_fixed} 轮")
    print(f"新自适应策略     : {steps_adaptive} 轮")
    if steps_fixed > 0 and steps_adaptive < MAX_ITER:
        improvement = (steps_fixed - steps_adaptive) / steps_fixed * 100
        print(f"性能提升         : {improvement:.1f}%")
    else:
        print("性能提升         : N/A (自适应策略未在500轮内收敛)")
    print("="*50)

    # ===== 绘图 =====
    # 获取初始状态（第0轮）
    initial_states = sim_fixed.get_state_history()[0]
    global_mean = np.mean(initial_states)

    plt.figure(figsize=(10, 6))
    plt.plot(avg_fixed, '--', label=f'Fixed β=2.0 ({steps_fixed} steps)', linewidth=1.5)
    plt.plot(avg_adaptive, '-', label=f'Diff-Adaptive (βₘₐₓ=0.5, {steps_adaptive} steps)', linewidth=2.5)
    plt.xlabel('Iteration')
    plt.ylabel('Average State')
    plt.title(f'Consensus Convergence: Ring Topology (N={N_AGENTS})')
    plt.axhline(y=global_mean, color='r', linestyle=':', label='Global Mean')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('adaptive_vs_fixed_comparison_FINAL.png', dpi=200)
    plt.show()

if __name__ == "__main__":
    main()