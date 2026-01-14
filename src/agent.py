# src/agent.py
from .strategies import DeGrootStrategy, StubbornStrategy, SusceptibleStrategy

class Agent:
    def __init__(self, agent_id, initial_state, neighbors=None, strategy='deGroot', **strategy_params):
        self.id = agent_id
        self.state = initial_state
        self.next_state = None
        
        # 修复：正确处理neighbors参数
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
            self.state, neighbor_states
        )
    
    def set_neighbors(self, neighbors):
        """设置邻居列表"""
        self.neighbors = neighbors
    
    def commit_update(self):
        """将计算好的next_state正式更新为当前state"""
        if self.next_state is not None:
            self.state = self.next_state
            self.next_state = None
        else:
            print(f"⚠️ Agent {self.id}: next_state为None!")
    def get_strategy_info(self):
        """获取策略信息"""
        return self.strategy.get_description()