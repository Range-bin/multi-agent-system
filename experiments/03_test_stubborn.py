# experiments/03_test_stubborn.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

def test_stubborn_basic():
    """测试固执型策略基础功能"""
    print("=" * 60)
    print("固执型策略基础测试")
    print("=" * 60)
    
    # 测试不同α值
    alpha_values = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    for alpha in alpha_values:
        print(f"\n>>> 测试 α = {alpha}")
        
        # 创建模拟器，使用固执型策略
        sim = ConsensusSimulator(
            n_agents=10,
            topology='ring',
            initial_state_range=(0, 100),
            strategy='stubborn',
            strategy_params={'alpha': alpha}
        )
        
        # 运行仿真
        iterations = sim.run_until_convergence(max_iterations=200, tolerance=1e-4)
        
        # 分析结果
        metrics = sim.get_convergence_metrics()
        if metrics:
            print(f"  收敛轮数: {iterations}")
            print(f"  共识值: {metrics['consensus_value']:.4f}")
            print(f"  初始平均值: {metrics['initial_average']:.4f}")
            print(f"  偏差: {metrics['bias']:.4f}")
            
            # 验证固执度影响
            if alpha == 0:
                print(f"  ✅ α=0应等同于DeGroot，偏差应接近0")
            elif alpha == 1.0:
                print(f"  ✅ α=1应完全不收敛，迭代次数=200")
    
    print("\n" + "=" * 60)
    print("固执型策略验证标准:")
    print("1. α越大 → 收敛越慢")
    print("2. α越大 → 共识值越偏离初始平均值")
    print("3. α=1 → 完全不收敛（状态不变）")
    print("=" * 60)

if __name__ == '__main__':
    test_stubborn_basic()

def debug_strategy_calculation():
    """调试策略计算"""
    print("\n=== 策略计算验证 ===")
    
    from src.strategies import StubbornStrategy
    
    # 测试固执型策略计算
    strategy = StubbornStrategy(alpha=0.7)
    self_state = 37.45
    neighbor_states = [95.07, 73.20]  # 示例邻居
    
    result = strategy.compute_next_state(self_state, neighbor_states)
    print(f"自身状态: {self_state}")
    print(f"邻居状态: {neighbor_states}, 平均: {sum(neighbor_states)/len(neighbor_states):.2f}")
    print(f"α=0.7计算结果: {result:.2f}")
    print(f"理论: 0.7*37.45 + 0.3*84.14 = {0.7*37.45 + 0.3*84.14:.2f}")

def test_stubborn_with_details():
    """详细测试固执型策略"""
    print("\n>>> 详细测试 α = 0.7")
    
    sim = ConsensusSimulator(
        n_agents=5,  # 用少量智能体便于观察
        topology='ring',
        initial_state_range=(0, 100),
        strategy='stubborn',
        strategy_params={'alpha': 0.7}
    )
    
    iterations = sim.run_until_convergence(max_iterations=50, tolerance=1e-2)
    
    # 打印所有智能体最终状态
    history = sim.get_state_history()
    print(f"\n最终状态:")
    for i in range(sim.n_agents):
        print(f"Agent {i}: {history[0][i]:.2f} → {history[-1][i]:.2f}")
    
    print(f"\n标准差变化: {np.std(history[0]):.2f} → {np.std(history[-1]):.2f}")
    print(f"平均值变化: {np.mean(history[0]):.2f} → {np.mean(history[-1]):.2f}")

if __name__ == '__main__':
    debug_strategy_calculation()