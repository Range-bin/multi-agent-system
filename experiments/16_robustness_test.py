# experiments/16_robustness_test.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.consensus_simulator import ConsensusSimulator
from src.strategies import (
    SusceptibleStrategy,
    DiffAdaptiveStrategy,
    RobustDiffAdaptiveStrategy,
    NoiseResilientStrategy
)

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

    # 定义四种策略
    fixed_strategy = SusceptibleStrategy(beta=2.0)
    adaptive_strategy = DiffAdaptiveStrategy(beta_max=0.7, k=0.05)
    robust_adaptive_strategy = RobustDiffAdaptiveStrategy(beta_max=0.7, k=0.05, tau=30)
    noise_resilient_strategy = NoiseResilientStrategy(
        beta_max=0.6,
        k=0.05,
        tau=30,
        smoothing_window=3,
        trust_threshold=5.0
    )

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

        # 鲁棒自适应策略
        sim_robust = create_simulator_with_strategy(
            N_AGENTS, TOPOLOGY, INITIAL_RANGE, robust_adaptive_strategy, MAX_ITER
        )
        steps_robust = sim_robust.run_until_convergence(
            tolerance=TOLERANCE, 
            noise_std=params['noise_std'], 
            verbose=False
        )
        avg_robust = [np.mean(states) for states in sim_robust.get_state_history()]

        # 新增：抗噪增强策略
        sim_noise_resilient = create_simulator_with_strategy(
            N_AGENTS, TOPOLOGY, INITIAL_RANGE, noise_resilient_strategy, MAX_ITER
        )
        steps_noise_resilient = sim_noise_resilient.run_until_convergence(
            tolerance=TOLERANCE, 
            noise_std=params['noise_std'], 
            verbose=False
        )
        avg_noise_resilient = [np.mean(states) for states in sim_noise_resilient.get_state_history()]

        results[scenario_name] = {
            'fixed': steps_fixed,
            'adaptive': steps_adaptive,
            'robust': steps_robust,
            'noise_resilient': steps_noise_resilient
        }
        histories[scenario_name] = {
            'fixed': avg_fixed,
            'adaptive': avg_adaptive,
            'robust': avg_robust,
            'noise_resilient': avg_noise_resilient,
            'global_mean': np.mean(sim_fixed.get_state_history()[0])  # 初始全局均值
        }

    # ===== 结果输出 =====
    print("\n" + "="*70)
    print("📊 鲁棒性测试结果汇总")
    print("="*70)
    for scenario, res in results.items():
        print(f"\n【{scenario}】")
        print(f"  固定策略           : {res['fixed']} 轮")
        print(f"  自适应策略         : {res['adaptive']} 轮")
        print(f"  鲁棒自适应策略     : {res['robust']} 轮")
        print(f"  抗噪增强策略       : {res['noise_resilient']} 轮")
        if res['fixed'] < MAX_ITER and res['adaptive'] < MAX_ITER:
            improvement = (res['fixed'] - res['adaptive']) / res['fixed'] * 100
            print(f"  性能提升 (vs Fixed) : {improvement:.1f}%")

    # ===== 绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for idx, (scenario, hist) in enumerate(histories.items()):
        ax = axes[idx]
        ax.plot(hist['fixed'], '--', label=f'Fixed β=2.0 ({results[scenario]["fixed"]} steps)', linewidth=1.5)
        ax.plot(hist['adaptive'], '-', label=f'Diff-Adaptive ({results[scenario]["adaptive"]} steps)', linewidth=2)
        ax.plot(hist['robust'], '-.', label=f'Robust-Adaptive ({results[scenario]["robust"]} steps)', linewidth=2)
        ax.plot(hist['noise_resilient'], ':', label=f'Noise-Resilient ({results[scenario]["noise_resilient"]} steps)', linewidth=2.5)
        ax.axhline(y=hist['global_mean'], color='r', linestyle=':', label='Global Mean')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Average State', fontsize=12)
        ax.set_title(f'{scenario} (Noise σ={scenarios[scenario]["noise_std"]})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('robustness_comparison_all.png', dpi=200, bbox_inches='tight')
    plt.show()

    print("\n✅ 鲁棒性测试完成！结果已保存至 'robustness_comparison_all.png'")

if __name__ == "__main__":
    main()