# experiments/deep_test.py
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

print("=" * 60)
print("深入测试：固执型策略的共识值偏差")
print("=" * 60)

# 测试不同拓扑和α值
test_cases = [
    ('ring', '环形网络', 0.7),
    ('ring', '环形网络', 0.9),
    ('star', '星型网络', 0.7),
    ('star', '星型网络', 0.9),
]

for topology, name, alpha in test_cases:
    print(f"\n>>> {name}, α={alpha}")
    
    sim = ConsensusSimulator(
        n_agents=5,
        topology=topology,
        initial_state_range=(0, 100),
        strategy='stubborn',
        strategy_params={'alpha': alpha}
    )
    
    # 运行更多轮次，更严格阈值
    iterations = sim.run_until_convergence(max_iterations=300, tolerance=1e-5)
    history = sim.get_state_history()
    
    consensus = np.mean(history[-1])
    initial_avg = np.mean(history[0])
    bias = consensus - initial_avg
    
    print(f"  收敛轮数: {iterations}")
    print(f"  共识值: {consensus:.6f}")
    print(f"  初始平均: {initial_avg:.6f}")
    print(f"  偏差: {bias:.6f}")
    print(f"  相对偏差: {abs(bias/initial_avg)*100:.4f}%")
    
    # 检查是否所有智能体状态相同
    final_std = np.std(history[-1])
    print(f"  最终标准差: {final_std:.6f}")