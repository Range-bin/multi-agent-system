# experiments/final_debug.py
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

print("=" * 60)
print("最终调试：3节点全连接网络")
print("=" * 60)

# 创建模拟器（会显示调试信息）
sim = ConsensusSimulator(n_agents=3, topology='complete', initial_state_range=(0, 10))

print(f"\n=== 运行共识算法 ===")
print(f"初始状态: {sim.state_history[0]}")
print(f"初始标准差: {np.std(sim.state_history[0]):.4f}")

# 运行一轮迭代，检查内部过程
print(f"\n--- 第1轮迭代 ---")
sim.run_iteration_deGroot()
history = sim.get_state_history()
print(f"第1轮后状态: {history[-1]}")
print(f"第1轮后标准差: {np.std(history[-1]):.4f}")

# 理论计算验证
print(f"\n=== 理论计算验证 ===")
states = sim.state_history[0]
print(f"智能体0应更新为: ({states[0]} + {states[1]} + {states[2]}) / 3 = {(states[0]+states[1]+states[2])/3:.4f}")
print(f"智能体1应更新为: ({states[1]} + {states[0]} + {states[2]}) / 3 = {(states[1]+states[0]+states[2])/3:.4f}")
print(f"智能体2应更新为: ({states[2]} + {states[0]} + {states[1]}) / 3 = {(states[2]+states[0]+states[1])/3:.4f}")

print(f"\n=== 检查状态更新 ===")
for i in range(3):
    print(f"Agent {i}: {sim.state_history[0][i]:.4f} -> {sim.state_history[1][i]:.4f}")