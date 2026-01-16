# strategies.py
from abc import ABC, abstractmethod

class ConsensusStrategy(ABC):
    """共识策略抽象基类"""
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def compute_next_state(self, self_state, neighbor_states):
        pass

class DeGrootStrategy(ConsensusStrategy):
    """标准DeGroot共识策略：取自身与所有邻居状态的平均值"""
    def compute_next_state(self, self_state, neighbor_states):
        if not neighbor_states:
            return self_state
        total = self_state + sum(neighbor_states)
        return total / (1 + len(neighbor_states))

class StubbornStrategy(ConsensusStrategy):
    """固执型策略：保留部分自身状态，混合邻居平均
    x_i(t+1) = alpha * x_i(t) + (1 - alpha) * avg(neighbors)
    alpha ∈ [0, 1]，alpha越大越固执
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha 必须在 [0, 1] 范围内")
        self.alpha = alpha

    def compute_next_state(self, self_state, neighbor_states):
        if not neighbor_states:
            return self_state
        neighbor_avg = sum(neighbor_states) / len(neighbor_states)
        return self.alpha * self_state + (1 - self.alpha) * neighbor_avg

class SusceptibleStrategy(ConsensusStrategy):
    """易受影响型策略（保留原始公式框架）
    公式：x_i(t+1) = (1/β) * x_i(t) + ((β - 1)/β) * avg(neighbors)
    
    注意：
    - 当 β = 1.0 时，原公式退化为 x_i(t+1) = x_i(t)，不合理
    - 因此特别处理：β = 1.0 时，退化为标准 DeGroot 行为
    - 当 β > 1.0 时，自身权重 > 邻居权重 → 实际表现为“弱固执”
      （论文中可解释为：β 控制“信息采纳保守程度”，β≥1 表示个体至少保留部分自我信念）
    """
    def __init__(self, beta=1.0):
        super().__init__()
        if beta < 1.0:
            raise ValueError("beta 必须 >= 1.0（按实验设计约束）")
        self.beta = beta

    def compute_next_state(self, self_state, neighbor_states):
        if not neighbor_states:
            return self_state
        
        # 关键修复：β=1.0 时，严格使用 DeGroot 更新规则
        if self.beta == 1.0:
            total = self_state + sum(neighbor_states)
            return total / (1 + len(neighbor_states))
        
        # 保留原始公式（β > 1.0）
        neighbor_avg = sum(neighbor_states) / len(neighbor_states)
        self_weight = 1.0 / self.beta
        neighbor_weight = (self.beta - 1.0) / self.beta
        return self_weight * self_state + neighbor_weight * neighbor_avg