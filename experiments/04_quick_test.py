# experiments/minimal_test.py
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agent import Agent

print("=== 最小化测试：Agent + 固执型策略 ===")

# 创建固执型Agent
agent = Agent(
    agent_id=0,
    initial_state=37.45,
    neighbors=[1, 2],
    strategy='stubborn',
    alpha=0.7
)

print(f"1. Agent创建检查:")
print(f"  策略类型 = {agent.strategy.__class__.__name__}")
print(f"  α参数 = {agent.strategy.alpha}")
print(f"  邻居 = {agent.neighbors}")

# 手动调用策略计算
neighbor_states = [95.07, 73.20]
print(f"\n2. 策略计算测试:")
print(f"  输入: 自身状态={agent.state}")
print(f"  输入: 邻居状态={neighbor_states}")
print(f"  邻居平均值 = {np.mean(neighbor_states):.2f}")

# 现在compute_next_state会返回值
result = agent.compute_next_state(neighbor_states)
print(f"  计算结果: {result:.2f}")

# 验证
expected = 0.7 * 37.45 + 0.3 * np.mean(neighbor_states)
print(f"\n3. 验证:")
print(f"  理论: 0.7×37.45 + 0.3×84.135 = {expected:.2f}")
print(f"  实际: {result:.2f}")
print(f"  是否匹配: {abs(result - expected) < 0.01}")

# 检查next_state是否设置
print(f"\n4. 检查next_state:")
print(f"  agent.next_state = {agent.next_state:.2f}")

# 测试commit_update
print(f"\n5. 状态更新测试:")
old_state = agent.state
agent.commit_update()
print(f"  更新前: {old_state:.2f}")
print(f"  更新后: {agent.state:.2f}")
print(f"  是否更新为计算结果: {abs(agent.state - result) < 0.01}")