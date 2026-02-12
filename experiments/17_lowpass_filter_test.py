# experiments/17_lowpass_filter_test.py

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
from src.strategies import LowPassFilterStrategy, SusceptibleStrategy

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
    print("🔍 启动低通滤波策略专项测试：高噪声下的鲁棒性验证...\n")
    
    N_AGENTS = 20
    TOPOLOGY = 'ring'
    INITIAL_RANGE = (0, 100)
    MAX_ITER = 1000
    TOLERANCE = 1e-3
    NOISE_STD = 2.0  # 强噪声

    # === 定义待测策略 ===
    fixed_strategy = SusceptibleStrategy(beta=2.0)  # 作为基线
    low_pass_strategy = LowPassFilterStrategy(
        alpha=0.9,      # 高平滑系数 → 强抗噪
        beta_max=0.6,
        k=0.05,
        tau=50
    )

    scenarios = {
        "无噪声": {"noise_std": 0.0},
        "有噪声": {"noise_std": NOISE_STD}
    }

    results = {}
    histories = {}

    for scenario_name, params in scenarios.items():
        print(f"▶ 测试场景: {scenario_name} (噪声σ={params['noise_std']})...")
        
        # 基线策略
        sim_fixed = create_simulator_with_strategy(
            N_AGENTS, TOPOLOGY, INITIAL_RANGE, fixed_strategy, MAX_ITER
        )
        steps_fixed = sim_fixed.run_until_convergence(
            tolerance=TOLERANCE, 
            noise_std=params['noise_std'], 
            verbose=False
        )
        avg_fixed = [np.mean(states) for states in sim_fixed.get_state_history()]
        
        # 低通滤波策略
        sim_lowpass = create_simulator_with_strategy(
            N_AGENTS, TOPOLOGY, INITIAL_RANGE, low_pass_strategy, MAX_ITER
        )
        steps_lowpass = sim_lowpass.run_until_convergence(
            tolerance=TOLERANCE, 
            noise_std=params['noise_std'], 
            verbose=False
        )
        avg_lowpass = [np.mean(states) for states in sim_lowpass.get_state_history()]

        results[scenario_name] = {
            'fixed': steps_fixed,
            'lowpass': steps_lowpass
        }
        histories[scenario_name] = {
            'fixed': avg_fixed,
            'lowpass': avg_lowpass,
            'global_mean': np.mean(sim_fixed.get_state_history()[0])
        }

    # ===== 结果输出 =====
    print("\n" + "="*60)
    print("📊 低通滤波策略专项测试结果")
    print("="*60)
    for scenario, res in results.items():
        print(f"\n【{scenario}】")
        print(f"  固定策略 (β=2.0) : {res['fixed']} 轮")
        print(f"  低通滤波策略     : {res['lowpass']} 轮")
        if res['fixed'] < MAX_ITER and res['lowpass'] < MAX_ITER:
            improvement = (res['fixed'] - res['lowpass']) / res['fixed'] * 100
            print(f"  性能提升 (vs Fixed) : {improvement:.1f}%")

    # ===== 绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for idx, (scenario, hist) in enumerate(histories.items()):
        ax = axes[idx]
        ax.plot(hist['fixed'], '--', label=f'Fixed β=2.0 ({results[scenario]["fixed"]} steps)', linewidth=2)
        ax.plot(hist['lowpass'], '-', label=f'Low-Pass Filter ({results[scenario]["lowpass"]} steps)', linewidth=2.5)
        ax.axhline(y=hist['global_mean'], color='r', linestyle=':', label='Global Mean')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average State')
        ax.set_title(f'{scenario} (Noise σ={scenarios[scenario]["noise_std"]})')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('lowpass_filter_test.png', dpi=200, bbox_inches='tight')
    plt.show()

    print("\n✅ 低通滤波策略测试完成！结果已保存至 'lowpass_filter_test.png'")

if __name__ == "__main__":
    main()