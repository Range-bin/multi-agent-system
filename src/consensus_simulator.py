# src/consensus_simulator.py
import numpy as np
from .agent import Agent
from .network_generator import generate_topology, get_adjacency_list

class ConsensusSimulator:
    def __init__(self, n_agents=10, topology='complete', initial_state_range=(0, 100), **kwargs):
        """
        初始化模拟器。
        """
        self.n_agents = n_agents
        
        # 1. 生成网络
        self.G = generate_topology(topology, n_agents, **kwargs)
        self.adj_list = get_adjacency_list(self.G)
        
        print(f"\n=== 网络生成调试 ===")
        print(f"网络类型: {topology}")
        print(f"节点数: {n_agents}")
        print(f"邻接列表前3个节点: {dict(list(self.adj_list.items())[:3])}")
        for i in range(min(3, n_agents)):
            print(f"  Agent {i}: 邻居 = {self.adj_list.get(i, [])}")
        
        # 2. 创建智能体
        self.agents = {}
        np.random.seed(42)  # 固定随机种子，使结果可复现
        initial_states = np.random.uniform(initial_state_range[0], initial_state_range[1], n_agents)
        
        print(f"\n=== 智能体创建调试 ===")
        for i in range(n_agents):
            neighbors = self.adj_list.get(i, [])
    
            self.agents[i] = Agent(
                agent_id=i,
                initial_state=initial_states[i],
                neighbors=neighbors,  # 关键：直接传递
                strategy='deGroot',   # 使用默认策略
                # 如果有其他策略参数，在这里添加
            )        
            # 验证传递是否正确
            if i < 3:  # 只检查前3个
                print(f"Agent {i}:")
                print(f"  传递的邻居列表 = {neighbors}")
                print(f"  对象存储的邻居 = {self.agents[i].neighbors}")
                print(f"  两者是否相等: {neighbors == self.agents[i].neighbors}")
            
        # 3. 记录状态历史
        self.state_history = [initial_states.copy()]
    
    def run_iteration_deGroot(self):
        """
        执行一轮DeGroot共识迭代。
        """
        # 步骤1：所有智能体计算下一状态
        for agent_id, agent in self.agents.items():
            print(f"\n  Agent {agent_id}:")
            print(f"    当前状态: {agent.state}")
            print(f"    邻居: {agent.neighbors}")            
            if not agent.neighbors:
                print(f"    ⚠️ 无邻居，保持原状态")
                agent.next_state = agent.state
                continue
        
            # 获取邻居状态
            neighbor_states = []
            for neighbor_id in agent.neighbors:
                neighbor_state = self.agents[neighbor_id].state
                neighbor_states.append(self.agents[neighbor_id].state)
                print(f"    邻居{neighbor_id}状态: {neighbor_state}")
            
            # 计算平均值（包括自身）
            total = agent.state + sum(neighbor_states)
            count = 1 + len(neighbor_states)
            agent.next_state = total / count

            print(f"    计算: ({agent.state} + {sum(neighbor_states)}) / {count} = {agent.next_state}")
        # 步骤2：所有智能体统一更新状态
        print(f"\n  [提交更新]")
        new_states = []
        for agent_id, agent in self.agents.items():
            old_state = agent.state
            agent.commit_update()
            new_states.append(agent.state)
            print(f"    Agent {agent_id}: {old_state:.4f} -> {agent.state:.4f}")
        # 记录历史
        self.state_history.append(np.array(new_states))
        print(f"  本轮标准差: {np.std(new_states):.6f}")
    
    def run_until_convergence(self, max_iterations=500, tolerance=1e-5):
        """
        运行仿真直到达成共识或达到最大迭代次数。
        """
        print(f"初始标准差: {np.std(self.state_history[-1]):.6f}")
        
        for iteration in range(max_iterations):
            self.run_iteration_deGroot()
            
            current_states = self.state_history[-1]
            std_dev = np.std(current_states)
            
            # 每10轮或前5轮显示一次进度
            if iteration < 5 or (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration+1}: 标准差 = {std_dev:.6f}")
                if iteration < 3:
                    print(f"  当前状态（前3个）: {current_states[:3]}")
            
            # 收敛判定：标准差小于阈值
            if std_dev < tolerance:
                print(f"✅ 共识在 {iteration+1} 轮后达成。最终标准差: {std_dev:.6f}")
                print(f"最终共识值: {np.mean(current_states):.4f}")
                return iteration + 1
        
        final_std = np.std(self.state_history[-1])
        print(f"⚠️ 在 {max_iterations} 轮后未完全达成共识。最终标准差: {final_std:.6f}")
        return max_iterations
    
    def get_state_history(self):
        """返回状态历史，形状为 (迭代次数, n_agents) 的numpy数组"""
        return np.array(self.state_history)