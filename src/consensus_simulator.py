# src/consensus_simulator.py
import numpy as np
from .agent import Agent
from .network_generator import generate_topology, get_adjacency_list

class ConsensusSimulator:
    def __init__(self, n_agents=10, topology='complete', initial_state_range=(0, 100),
                 strategy='deGroot', strategy_params=None, **kwargs):
        """
        初始化共识模拟器。
        
        参数:
            n_agents (int): 智能体数量
            topology (str): 网络类型 ('ring', 'star', 'complete')
            initial_state_range (tuple): 初始状态范围 (min, max)
            strategy (str): 共识策略 ('deGroot', 'stubborn', 'susceptible')
            strategy_params (dict): 策略参数，如 {'alpha': 0.7}
        """
        self.n_agents = n_agents
        
        # 1. 生成网络拓扑
        self.G = generate_topology(topology, n_agents, **kwargs)
        self.adj_list = get_adjacency_list(self.G)
        
        print(f"\n=== 模拟器初始化 ===")
        print(f"网络类型: {topology}, 智能体数: {n_agents}")
        print(f"策略: {strategy}, 参数: {strategy_params}")
        
        # 2. 创建智能体
        self.agents = {}
        np.random.seed(42)  # 固定随机种子确保可复现
        initial_states = np.random.uniform(initial_state_range[0], initial_state_range[1], n_agents)
        
        for i in range(n_agents):
            neighbors = self.adj_list.get(i, [])
            if not neighbors:
                print(f"⚠️ 警告: Agent {i} 无邻居，将保持初始状态不变。")
            
            self.agents[i] = Agent(
                agent_id=i,
                initial_state=initial_states[i],
                neighbors=neighbors,
                strategy=strategy,
                **(strategy_params or {})
            )
            
            # 打印前两个智能体信息用于调试
            if i < 2:
                print(f"Agent {i}: 初始状态={initial_states[i]:.2f}, "
                      f"邻居数={len(neighbors)}, "
                      f"策略={self.agents[i].strategy.__class__.__name__}")

        # 3. 记录状态历史
        self.state_history = [initial_states.copy()]
        self._convergence_window = []  # 用于滑动窗口检测

    def run_iteration(self):
        """执行一轮共识迭代"""
        # 步骤1：计算下一状态
        for agent_id, agent in self.agents.items():
            if not agent.neighbors:
                agent.next_state = agent.state
                continue
            
            neighbor_states = [self.agents[j].state for j in agent.neighbors]
            agent.compute_next_state(neighbor_states)

        # 步骤2：统一更新状态
        new_states = []
        for agent in self.agents.values():
            agent.commit_update()
            new_states.append(agent.state)

        self.state_history.append(np.array(new_states))
        return np.std(new_states)

    def _is_converged_stable(self, current_std, tolerance=1e-6, window_size=3):
        """
        智能收敛判定：
        - 若当前标准差已低于阈值 → 立即返回 True（适用于缓慢收敛）
        - 否则使用滑动窗口确保稳定性
        """
        if current_std < tolerance:
            return True

        # 滑动窗口后备机制
        self._convergence_window.append(current_std)
        if len(self._convergence_window) > window_size:
            self._convergence_window.pop(0)
        
        if len(self._convergence_window) < window_size:
            return False
        
        if any(std >= tolerance for std in self._convergence_window):
            return False
        
        last_change = abs(self._convergence_window[-1] - self._convergence_window[-2])
        return last_change < tolerance * 0.1

    def _detect_oscillation(self, current_std, tolerance=1e-6, lookback=10):
        """
        检测状态震荡：若最近几轮标准差出现“先降后升”，且仍高于阈值，则提前终止
        """
        if len(self.state_history) < lookback + 1:
            return False
        
        recent_stds = [np.std(hist) for hist in self.state_history[-lookback:]]
        min_recent = min(recent_stds)
        
        # 如果当前标准差比近期最小值大50%以上，且仍高于容忍度，视为震荡
        return current_std > min_recent * 1.5 and current_std > tolerance

    def run_until_convergence(self, max_iterations=1000, tolerance=1e-6, verbose=True):
        """
        运行仿真直到收敛或达到最大迭代次数。
        
        改进：
        - 默认 max_iterations=1000（原500）
        - 默认 tolerance=1e-6（原1e-5）
        - 新增震荡检测
        """
        initial_std = np.std(self.state_history[-1])
        if verbose:
            print(f"初始标准差: {initial_std:.6f}")
            print(f"初始平均值: {np.mean(self.state_history[-1]):.4f}")
        
        self._convergence_window = []

        for iteration in range(max_iterations):
            std_dev = self.run_iteration()
            
            # 显示进度（前5轮 + 每100轮）
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

            # 震荡检测（运行50轮后启用）
            if iteration > 50 and self._detect_oscillation(std_dev, tolerance=tolerance):
                if verbose:
                    print(f"⚠️ 检测到状态震荡，提前终止。当前标准差: {std_dev:.6f}")
                break

        # 达到最大迭代次数
        final_std = np.std(self.state_history[-1])
        if verbose:
            print(f"❌ 在 {max_iterations} 轮后未达成共识。最终标准差: {final_std:.6f}")
        return max_iterations

    def get_state_history(self):
        """返回状态历史数组 (iterations, n_agents)"""
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