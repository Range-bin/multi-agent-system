# experiments/05_test_fixed_simulator.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt  # 添加这行

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

print("=" * 60)
print("修复验证：固执型策略在仿真中")
print("=" * 60)

# 测试1：DeGroot作为基准
print("\n1. DeGroot策略 (基准):")
sim1 = ConsensusSimulator(
    n_agents=5,
    topology='ring',
    initial_state_range=(0, 100),
    strategy='deGroot'
)
iter1 = sim1.run_until_convergence(max_iterations=50, tolerance=1e-2)
history1 = sim1.get_state_history()
print(f"  收敛轮数: {iter1}")
print(f"  共识值: {np.mean(history1[-1]):.4f}")
print(f"  初始平均: {np.mean(history1[0]):.4f}")

# 测试2：固执型策略 α=0.7
print("\n2. 固执型策略 α=0.7:")
sim2 = ConsensusSimulator(
    n_agents=5,
    topology='ring',
    initial_state_range=(0, 100),
    strategy='stubborn',
    strategy_params={'alpha': 0.7}
)
iter2 = sim2.run_until_convergence(max_iterations=50, tolerance=1e-2)
history2 = sim2.get_state_history()
print(f"  收敛轮数: {iter2}")
print(f"  共识值: {np.mean(history2[-1]):.4f}")
print(f"  初始平均: {np.mean(history2[0]):.4f}")

# 关键验证：两者共识值应该不同！
print("\n3. 关键验证:")
print(f"  DeGroot共识值: {np.mean(history1[-1]):.4f}")
print(f"  固执型共识值: {np.mean(history2[-1]):.4f}")
diff = abs(np.mean(history1[-1]) - np.mean(history2[-1]))
print(f"  差值: {diff:.4f}")
print(f"  是否不同: {diff > 0.01}")

# 打印最终状态对比
print("\n4. 最终状态对比 (前3个智能体):")
print(f"  DeGroot: {history1[-1][:3]}")
print(f"  固执型: {history2[-1][:3]}")