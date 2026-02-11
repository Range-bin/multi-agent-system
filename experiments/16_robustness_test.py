# experiments/16_robustness_test.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator
from src.strategies import DiffAdaptiveStrategy, SusceptibleStrategy

def create_simulator_with_strategy(n_agents, topology, initial_state_range, strategy_instance, max_iterations):
    """通用创建函数，用于注入自定义策略"""
    sim = ConsensusSimulator(
        n_agents=n_agents,
        topology=topology,
        initial_state_range=initial_state_range,
        strategy='deGroot',  # 临时占位
        max_iterations=max_iterations,
        verbose=False
    )
    for agent in sim.agents.values():
        agent.strategy = strategy_instance
    return sim

def main():
    print("🛡️ 启动鲁棒性测试实验：通信噪声下的性能对比...\n")
    
    N_AGENTS = 20
    TOPOLOGY = 'ring'
    INITIAL_RANGE = (0, 100)
    MAX_ITER = 1000
    TOLERANCE = 1e-3  # 噪声下放宽收敛条件
    NOISE_STD = 2.0   # 通信噪声标准差

    # 定义策略
    fixed_strategy = SusceptibleStrategy(beta=2.0)
    adaptive_strategy = DiffAdaptiveStrategy(beta_max=0.7, k=0.05)

    scenarios = {
        "无噪声": {"noise_std": 0.0},
        "有噪声": {"noise_std": NOISE_STD}
    }

    results = {}
    histories = {}

    for scenario_name, params in scenarios.items():
        print(f"▶ 测试场景: {scenario_name} (噪声σ={params['noise_std']})...")
        
        # 固定策略
        sim_fixed = create_simulator_with_strategy(
            N_AGENTS, TOPOLOGY, INITIAL_RANGE, fixed_strategy, MAX_ITER
        )
        steps_fixed = sim_fixed.run_until_convergence(
            tolerance=TOLERANCE, 
            noise_std=params['noise_std'], 
            verbose=False
        )
        avg_fixed = [np.mean(states) for states in sim_fixed.get_state_history()]
        
        # 自适应策略
        sim_adaptive = create_simulator_with_strategy(
            N_AGENTS, TOPOLOGY, INITIAL_RANGE, adaptive_strategy, MAX_ITER
        )
        steps_adaptive = sim_adaptive.run_until_convergence(
            tolerance=TOLERANCE, 
            noise_std=params['noise_std'], 
            verbose=False
        )
        avg_adaptive = [np.mean(states) for states in sim_adaptive.get_state_history()]

        results[scenario_name] = {
            'fixed': steps_fixed,
            'adaptive': steps_adaptive
        }
        histories[scenario_name] = {
            'fixed': avg_fixed,
            'adaptive': avg_adaptive,
            'global_mean': np.mean(sim_fixed.get_state_history()[0])  # 初始全局均值
        }

    # ===== 结果输出 =====
    print("\n" + "="*60)
    print("📊 鲁棒性测试结果汇总")
    print("="*60)
    for scenario, res in results.items():
        print(f"\n【{scenario}】")
        print(f"  固定策略       : {res['fixed']} 轮")
        print(f"  自适应策略     : {res['adaptive']} 轮")
        if res['fixed'] < MAX_ITER and res['adaptive'] < MAX_ITER:
            improvement = (res['fixed'] - res['adaptive']) / res['fixed'] * 100
            print(f"  性能提升       : {improvement:.1f}%")
        else:
            print(f"  性能提升       : N/A (未完全收敛)")

    # ===== 绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for idx, (scenario, hist) in enumerate(histories.items()):
        ax = axes[idx]
        ax.plot(hist['fixed'], '--', label=f'Fixed β=2.0 ({results[scenario]["fixed"]} steps)', linewidth=1.5)
        ax.plot(hist['adaptive'], '-', label=f'Diff-Adaptive ({results[scenario]["adaptive"]} steps)', linewidth=2.5)
        ax.axhline(y=hist['global_mean'], color='r', linestyle=':', label='Global Mean')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average State')
        ax.set_title(f'{scenario} (Noise σ={scenarios[scenario]["noise_std"]})')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('robustness_comparison_noise.png', dpi=200)
    plt.show()

    print("\n✅ 鲁棒性测试完成！结果已保存至 'robustness_comparison_noise.png'")

if __name__ == "__main__":
    main()