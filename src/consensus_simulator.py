# src/consensus_simulator.py

import numpy as np
from .network_generator import generate_topology, get_adjacency_list  # ← 修正导入
from .agent import Agent  # ← 注意：你的文件叫 agent.py，不是 agents.py

class ConsensusSimulator:
    def __init__(self, n_agents, topology='complete', initial_state_range=(0, 1), 
                 strategy='deGroot', strategy_params=None, max_iterations=1000, verbose=True):
        """
        初始化共识模拟器。
        参数:
            n_agents (int): 智能体数量。
            topology (str): 网络拓扑类型 ('complete', 'ring', 'star', 'small_world')。
            initial_state_range (tuple): 初始状态范围 (min, max)。
            strategy (str): 默认策略名称（会被 agent.strategy 覆盖）。
            strategy_params (dict): 策略参数。
            max_iterations (int): 最大迭代次数。
            verbose (bool): 是否打印详细信息。
        """
        self.n_agents = n_agents
        self.topology = topology
        self.initial_state_range = initial_state_range
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.state_history = []

        # 1. 生成网络拓扑
        self.G = generate_topology(topology, n_agents)
        self.adj_list = get_adjacency_list(self.G)

        if verbose:
            self._print_network_info()

        # 2. 初始化智能体
        self.agents = {}
        np.random.seed(42)  # 固定种子确保可复现
        initial_states = np.random.uniform(initial_state_range[0], initial_state_range[1], n_agents)
        for i in range(n_agents):
            neighbors = self.adj_list.get(i, [])
            if not neighbors and verbose:
                print(f"⚠️ 警告: Agent {i} 无邻居，将保持初始状态不变。")
            self.agents[i] = Agent(
                agent_id=i,
                initial_state=initial_states[i],
                neighbors=neighbors,
                strategy=strategy,
                **(strategy_params or {})
            )
        self.state_history = [initial_states.copy()]
        self._convergence_window = []

    def _print_network_info(self):
        print(f"=== 模拟器初始化 ===")
        print(f"网络类型: {self.topology}, 智能体数: {self.n_agents}")
        for i in range(min(3, self.n_agents)):
            print(f"  节点{i}的邻居: {self.adj_list.get(i, [])}")
        first_agent = self.agents[0]
        print(f"策略: {first_agent.strategy.__class__.__name__}, "
              f"参数: {getattr(first_agent.strategy, '__dict__', {})}")
        print(f"Agent 0: 初始状态={first_agent.state:.2f}, "
              f"邻居数={len(first_agent.neighbors)}, "
              f"策略={first_agent.strategy.__class__.__name__}")

    def get_state_history(self):
        return np.array(self.state_history)

    def run_iteration(self, noise_std=0.0):
        """执行一轮共识迭代"""
        new_states = []
        current_states = {i: agent.state for i, agent in self.agents.items()}

        for agent_id, agent in self.agents.items():
            if not agent.neighbors:
                new_states.append(agent.state)
                continue

            # 获取邻居当前状态
            neighbor_states = [current_states[j] for j in agent.neighbors]

            # 添加通信噪声
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, size=len(neighbor_states))
                neighbor_states = np.array(neighbor_states) + noise
                neighbor_states = neighbor_states.tolist()

            # 计算下一状态
            next_state = agent.strategy.compute_next_state(agent.state, neighbor_states)
            new_states.append(next_state)

        # 统一更新
        for i, state in enumerate(new_states):
            self.agents[i].state = state
        self.state_history.append(np.array(new_states))
        return np.std(new_states)

    def _is_converged_stable(self, std_dev, tolerance=1e-6, window_size=5):
        """稳定收敛判定"""
        self._convergence_window.append(std_dev)
        if len(self._convergence_window) > window_size:
            self._convergence_window.pop(0)
        return len(self._convergence_window) == window_size and all(s < tolerance for s in self._convergence_window)

    def _detect_oscillation(self, std_dev, tolerance=1e-6, window_size=10, threshold=0.5):
        """震荡检测"""
        if not hasattr(self, '_oscillation_window'):
            self._oscillation_window = []
        self._oscillation_window.append(std_dev)
        if len(self._oscillation_window) > window_size:
            self._oscillation_window.pop(0)
        if len(self._oscillation_window) == window_size:
            arr = np.array(self._oscillation_window)
            return (np.std(arr) / (np.mean(arr) + 1e-8)) > threshold
        return False

    def run_until_convergence(self, max_iterations=1000, tolerance=1e-6, noise_std=0.0, verbose=True):
        """运行直到收敛"""
        initial_std = np.std(self.state_history[-1])
        if verbose:
            print(f"初始标准差: {initial_std:.6f}")
            print(f"初始平均值: {np.mean(self.state_history[-1]):.4f}")

        self._convergence_window = []
        for iteration in range(max_iterations):
            std_dev = self.run_iteration(noise_std=noise_std)

            if verbose and (iteration < 5 or (iteration + 1) % 100 == 0):
                print(f"迭代 {iteration+1}: 标准差 = {std_dev:.6f}")

            # 收敛判定
            if self._is_converged_stable(std_dev, tolerance=tolerance):
                if verbose:
                    final_val = np.mean(self.state_history[-1])
                    init_val = np.mean(self.state_history[0])
                    print(f"✅ 共识在 {iteration+1} 轮后达成。最终标准差: {std_dev:.2e}")
                    print(f"最终共识值: {final_val:.4f} (初始平均: {init_val:.4f})")
                return iteration + 1

            # 震荡检测
            if iteration > 50 and self._detect_oscillation(std_dev, tolerance=tolerance):
                if verbose:
                    print(f"⚠️ 检测到状态震荡，提前终止。当前标准差: {std_dev:.6f}")
                break

        final_std = np.std(self.state_history[-1])
        if verbose:
            print(f"❌ 在 {max_iterations} 轮后未达成共识。最终标准差: {final_std:.6f}")
        return max_iterations