# src/consensus_simulator.py
import numpy as np
from .agent import Agent
from .network_generator import generate_topology, get_adjacency_list

class ConsensusSimulator:
    def __init__(self, n_agents=10, topology='complete', initial_state_range=(0, 100), 
                 strategy='deGroot', strategy_params=None, **kwargs):
        """
        初始化模拟器。
        参数新增:
            strategy: 策略类型 'deGroot', 'stubborn', 'susceptible'
            strategy_params: 策略参数字典，如 {'alpha': 0.8}
        """
        self.n_agents = n_agents
        
        # 1. 生成网络
        self.G = generate_topology(topology, n_agents, **kwargs)
        self.adj_list = get_adjacency_list(self.G)
        
        print(f"\n=== 模拟器初始化 ===")
        print(f"网络类型: {topology}, 智能体数: {n_agents}")
        print(f"策略: {strategy}, 参数: {strategy_params}")
        
        # 2. 创建智能体
        self.agents = {}
        np.random.seed(42)  # 固定随机种子，使结果可复现
        initial_states = np.random.uniform(initial_state_range[0], initial_state_range[1], n_agents)
        
        for i in range(n_agents):
            neighbors = self.adj_list.get(i, [])
            
            self.agents[i] = Agent(
                agent_id=i,
                initial_state=initial_states[i],
                neighbors=neighbors,
                strategy=strategy,  # ✅ 使用传入的策略类型
                **(strategy_params or {})  # ✅ 传递策略参数
            )
            
            # 调试信息（只显示前2个）
            if i < 2:
                print(f"Agent {i}: 初始状态={initial_states[i]:.2f}, "
                      f"邻居数={len(neighbors)}, "
                      f"策略={self.agents[i].strategy.__class__.__name__}")
        
        # 3. 记录状态历史
        self.state_history = [initial_states.copy()]
    
    def run_iteration(self):
        """
        执行一轮共识迭代（通用方法，支持所有策略）
        """
        # 步骤1：所有智能体计算下一状态
        for agent_id, agent in self.agents.items():
            if not agent.neighbors:
                agent.next_state = agent.state
                continue
            
            # 获取邻居状态
            neighbor_states = []
            for neighbor_id in agent.neighbors:
                neighbor_states.append(self.agents[neighbor_id].state)
            
            # ✅ 关键：使用策略计算下一状态
            agent.compute_next_state(neighbor_states)
        
        # 步骤2：所有智能体统一更新状态
        new_states = []
        for agent_id, agent in self.agents.items():
            agent.commit_update()
            new_states.append(agent.state)
        
        # 记录历史
        self.state_history.append(np.array(new_states))
        
        # 返回本轮标准差（用于调试）
        return np.std(new_states)
    
    def run_until_convergence(self, max_iterations=500, tolerance=1e-5, verbose=True):
        """
        运行仿真直到达成共识或达到最大迭代次数。
        """
        initial_std = np.std(self.state_history[-1])
        if verbose:
            print(f"初始标准差: {initial_std:.6f}")
            print(f"初始平均值: {np.mean(self.state_history[-1]):.4f}")
        
        for iteration in range(max_iterations):
            std_dev = self.run_iteration()  # ✅ 调用通用方法
            
            # 进度显示
            if verbose and (iteration < 5 or (iteration + 1) % 10 == 0):
                current_states = self.state_history[-1]
                print(f"迭代 {iteration+1}: 标准差 = {std_dev:.6f}")
                if iteration < 3:
                    print(f"  当前状态（前3个）: {current_states[:3]}")
            
            # 收敛判定：标准差小于阈值
            if std_dev < tolerance:
                if verbose:
                    print(f"✅ 共识在 {iteration+1} 轮后达成。最终标准差: {std_dev:.6f}")
                    final_value = np.mean(self.state_history[-1])
                    initial_value = np.mean(self.state_history[0])
                    print(f"最终共识值: {final_value:.4f} (初始平均: {initial_value:.4f})")
                return iteration + 1
        
        final_std = np.std(self.state_history[-1])
        if verbose:
            print(f"⚠️ 在 {max_iterations} 轮后未完全达成共识。最终标准差: {final_std:.6f}")
        return max_iterations
    
    def get_state_history(self):
        """返回状态历史，形状为 (迭代次数, n_agents) 的numpy数组"""
        return np.array(self.state_history)
    
    def get_convergence_metrics(self):
        """获取收敛性能指标"""
        if len(self.state_history) < 2:
            return None
        
        metrics = {
            'total_iterations': len(self.state_history) - 1,
            'final_std': np.std(self.state_history[-1]),
            'consensus_value': np.mean(self.state_history[-1]),
            'initial_average': np.mean(self.state_history[0]),
            'bias': np.mean(self.state_history[-1]) - np.mean(self.state_history[0]),
        }
        return metrics