# experiments/09_test_new_strategy.py
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入策略类
from src.strategies import SusceptibleStrategy

print("=== 测试新的易受影响型策略 ===")

# 测试权重计算
beta_values = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]

for beta in beta_values:
    try:
        strategy = SusceptibleStrategy(beta=beta)
        
        # 测试计算
        self_state = 50.0
        neighbor_states = [60.0, 70.0]  # 平均=65.0
        result = strategy.compute_next_state(self_state, neighbor_states)
        
        # 计算理论权重
        if beta == 1.0:
            print(f"β={beta}: DeGroot行为，结果={(50+60+70)/3:.2f}")
        else:
            self_weight = 1.0 / beta
            neighbor_weight = (beta - 1.0) / beta
            expected = self_weight * 50.0 + neighbor_weight * 65.0
            print(f"β={beta}: 自身权重={self_weight:.3f}, 邻居权重={neighbor_weight:.3f}")
            print(f"     结果={result:.2f}, 预期={expected:.2f}, 匹配={abs(result-expected)<0.01}")
            
    except ValueError as e:
        print(f"β={beta}: 错误 - {e}")