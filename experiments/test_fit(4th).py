# experiments/test_fix.py
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.agent import Agent

# 测试Agent初始化
print("测试Agent邻居传递:")
agent = Agent(agent_id=0, initial_state=5.0, neighbors=[1, 2, 3])
print(f"Agent 0邻居: {agent.neighbors}")  # 应该输出 [1, 2, 3]
print(f"邻居数量: {len(agent.neighbors)}")

# 测试无邻居情况
agent2 = Agent(agent_id=1, initial_state=10.0)
print(f"\nAgent 1邻居: {agent2.neighbors}")  # 应该输出 []