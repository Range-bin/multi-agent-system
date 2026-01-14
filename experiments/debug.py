# experiments/debug_test.py
import sys
import os
import numpy as np  # 添加这行！

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

# 测试1：简单的3节点全连接网络
print("=" * 60)
print("调试测试：3节点全连接网络")
print("=" * 60)

sim = ConsensusSimulator(n_agents=3, topology='complete', initial_state_range=(0, 10))

# 打印初始状态
print(f"初始状态: {sim.state_history[0]}")
print(f"初始标准差: {np.std(sim.state_history[0]):.4f}")

# 只运行3轮迭代
for i in range(3):
    print(f"\n--- 第{i+1}轮迭代 ---")
    sim.run_iteration_deGroot()
    history = sim.get_state_history()
    print(f"第{i+1}轮后状态: {history[-1]}")
    print(f"第{i+1}轮后标准差: {np.std(history[-1]):.4f}")

print("\n" + "=" * 60)
print("调试测试：3节点环形网络")
print("=" * 60)

sim2 = ConsensusSimulator(n_agents=3, topology='ring', initial_state_range=(0, 10))
print(f"初始状态: {sim2.state_history[0]}")
print(f"Agent 0邻居: {sim2.agents[0].neighbors}")
print(f"Agent 1邻居: {sim2.agents[1].neighbors}")
print(f"Agent 2邻居: {sim2.agents[2].neighbors}")

# 运行一轮
print("\n--- 执行一轮迭代 ---")
sim2.run_iteration_deGroot()
print(f"第1轮后状态: {sim2.get_state_history()[-1]}")