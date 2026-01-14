# src/strategies.py
"""
共识策略接口定义
为第二阶段：固执型策略和易受影响型策略做准备
"""

import numpy as np

class ConsensusStrategy:
    """共识策略基类（抽象接口）"""
    
    def __init__(self, **params):
        """
        初始化策略参数
        params: 策略特定参数，如固执度alpha、影响增益beta等
        """
        self.params = params
    
    def compute_next_state(self, self_state, neighbor_states):
        """
        计算下一时刻的状态
        参数:
            self_state: 自身当前状态
            neighbor_states: 邻居状态列表
        返回:
            下一时刻的状态值
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def get_description(self):
        """返回策略描述"""
        return f"{self.__class__.__name__} with params: {self.params}"


class DeGrootStrategy(ConsensusStrategy):
    """标准DeGroot共识策略（平均策略）"""
    
    def compute_next_state(self, self_state, neighbor_states):
        """
        DeGroot更新规则：自身与邻居的简单平均
        公式: x_i(t+1) = (x_i(t) + Σ_{j∈N_i} x_j(t)) / (1 + |N_i|)
        """
        if not neighbor_states:  # 如果没有邻居
            return self_state
        
        total = self_state
        for state in neighbor_states:
            total += state
        
        return total / (1 + len(neighbor_states))


class StubbornStrategy(ConsensusStrategy):
    """固执型策略"""
    
    def __init__(self, alpha=0.5):
        """
        参数:
            alpha: 固执度，取值范围[0, 1]
                  alpha=0: 完全不相信自己（等同于DeGroot）
                  alpha=1: 完全固执（状态不变）
        """
        super().__init__(alpha=alpha)
        self.alpha = alpha
    
    def compute_next_state(self, self_state, neighbor_states):
        """
        固执型更新规则
        公式: x_i(t+1) = α * x_i(t) + (1-α) * avg(邻居状态)
        """
        if not neighbor_states or self.alpha == 1.0:
            return self_state
        
        # 计算邻居平均值
        neighbor_avg = sum(neighbor_states) / len(neighbor_states)
        
        # 加权平均
        return self.alpha * self_state + (1 - self.alpha) * neighbor_avg


class SusceptibleStrategy(ConsensusStrategy):
    """易受影响型策略（从众策略）"""
    
    def __init__(self, beta=1.5):
        """
        参数:
            beta: 影响增益，取值范围[1, +∞)
                  beta=1: 等同于DeGroot
                  beta>1: 更易受邻居影响
        """
        super().__init__(beta=beta)
        self.beta = beta
    
    def compute_next_state(self, self_state, neighbor_states):
        """
        易受影响型更新规则
        公式: x_i(t+1) = (1/β) * x_i(t) + ((β-1)/β) * avg(邻居状态)
        """
        if not neighbor_states or self.beta == 1.0:
            return self_state
        
        # 计算邻居平均值
        neighbor_avg = sum(neighbor_states) / len(neighbor_states)
        
        # 权重分配
        self_weight = 1.0 / self.beta
        neighbor_weight = (self.beta - 1.0) / self.beta
        
        return self_weight * self_state + neighbor_weight * neighbor_avg