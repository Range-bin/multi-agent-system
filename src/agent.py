# src/agent.py
from .strategies import DeGrootStrategy, StubbornStrategy, SusceptibleStrategy

class Agent:
    def __init__(self, agent_id, initial_state, neighbors=None, strategy='deGroot', **strategy_params):
        """
        初始化一个智能体。
        参数:
            agent_id: 智能体唯一标识
            initial_state: 初始状态值
            neighbors: 邻居ID列表
            strategy: 共识策略类型，可选 'deGroot', 'stubborn', 'susceptible'
            strategy_params: 策略特定参数，如alpha、beta等
        """
        self.id = agent_id
        self.state = initial_state
        self.next_state = None
        
        # 设置邻居
        if neighbors is None:
            self.neighbors = []
        else:
            self.neighbors = list(neighbors)  # 确保是列表
        
        # 初始化策略
        self.strategy = self._create_strategy(strategy, strategy_params)
    
    def _create_strategy(self, strategy_type, params):
        """创建共识策略对象"""
        if strategy_type == 'deGroot':
            return DeGrootStrategy(**params)
        elif strategy_type == 'stubborn':
            return StubbornStrategy(**params)
        elif strategy_type == 'susceptible':
            return SusceptibleStrategy(**params)
        else:
            raise ValueError(f"未知的策略类型: {strategy_type}")
    
    def compute_next_state(self, neighbor_states):
        """
        使用策略计算下一状态
        参数:
            neighbor_states: 邻居状态值列表
        """
        self.next_state = self.strategy.compute_next_state(
            self.state, 
            neighbor_states
        )
        return self.next_state
    
    def set_neighbors(self, neighbors):
        """设置邻居列表"""
        self.neighbors = list(neighbors) if neighbors else []
    
    def commit_update(self):
        """将计算好的next_state正式更新为当前state"""
        if self.next_state is not None:
            self.state = self.next_state
            self.next_state = None
    
    def get_strategy_info(self):
        """获取策略信息"""
        return self.strategy.get_description()
    
    def set_strategy(self, strategy_type, **params):
        """动态切换策略"""
        self.strategy = self._create_strategy(strategy_type, params)