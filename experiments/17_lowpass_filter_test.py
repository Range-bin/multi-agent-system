# experiments/17_lowpass_filter_test.py (增强版)

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.consensus_simulator import ConsensusSimulator
from src.strategies import LowPassFilterStrategy, SusceptibleStrategy

def create_simulator_with_strategy(n_agents, topology, initial_state_range, strategy_instance, max_iterations):
    sim = ConsensusSimulator(
        n_agents=n_agents,
        topology=topology,
        initial_state_range=initial_state_range,
        strategy='deGroot',
        max_iterations=max_iterations,
        verbose=False
    )
    for agent in sim.agents.values():
        agent.strategy = strategy_instance
    return sim

def main():
    print("🔍 启动噪声鲁棒性边界测试：寻找策略崩溃阈值...\n")
    
    N_AGENTS = 20
    TOPOLOGY = 'ring'
    INITIAL_RANGE = (0, 100)
    MAX_ITER = 1000
    TOLERANCE = 1e-3

    # 测试多个噪声水平
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    strategies = {
        "Fixed (β=2.0)": SusceptibleStrategy(beta=2.0),
        "Low-Pass Filter": LowPassFilterStrategy(alpha=0.9, beta_max=0.6, k=0.05, tau=50)
    }

    results = {name: [] for name in strategies}
    final_stds = {name: [] for name in strategies}

    for noise_std in noise_levels:
        print(f"▶ 测试噪声 σ = {noise_std} ...")
        for name, strategy in strategies.items():
            sim = create_simulator_with_strategy(N_AGENTS, TOPOLOGY, INITIAL_RANGE, strategy, MAX_ITER)
            steps = sim.run_until_convergence(tolerance=TOLERANCE, noise_std=noise_std, verbose=False)
            final_std = np.std(sim.get_state_history()[-1])
            results[name].append(steps)
            final_stds[name].append(final_std)

    # ===== 输出结果 =====
    print("\n" + "="*80)
    print("📊 噪声鲁棒性边界测试结果")
    print("="*80)
    print(f"{'噪声σ':<8} {'策略':<20} {'迭代轮数':<10} {'最终标准差':<12}")
    print("-"*80)
    for i, noise in enumerate(noise_levels):
        for name in strategies:
            steps = results[name][i]
            std = final_stds[name][i]
            steps_str = str(steps) if steps < MAX_ITER else "∞"
            print(f"{noise:<8} {name:<20} {steps_str:<10} {std:<12.4f}")

    # ===== 绘图：最终标准差 vs 噪声 =====
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name in strategies:
        ax.plot(noise_levels, final_stds[name], 'o-', label=name, linewidth=2, markersize=6)
    ax.set_xlabel('Noise Standard Deviation (σ)')
    ax.set_ylabel('Final Consensus Standard Deviation')
    ax.set_title('Robustness Boundary: Final Std vs Noise Level')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('robustness_boundary.png', dpi=200)
    plt.show()

    print("\n✅ 噪声鲁棒性边界测试完成！结果已保存至 'robustness_boundary.png'")

if __name__ == "__main__":
    main()