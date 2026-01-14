# experiments/01_test_basic.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.consensus_simulator import ConsensusSimulator

def ensure_figures_dir():
    """确保 figures 目录存在"""
    figures_dir = os.path.join(project_root, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    return figures_dir

def test_basic_simulation():
    print("=" * 60)
    print("测试1：全连接网络，DeGroot共识")
    print("=" * 60)
    
    # 确保 figures 目录存在
    figures_dir = ensure_figures_dir()
    
    # 1. 创建模拟器
    sim = ConsensusSimulator(n_agents=10, topology='complete')
    
    # 2. 运行仿真
    convergence_rounds = sim.run_until_convergence(max_iterations=50, tolerance=1e-5)
    
    # 3. 获取数据并绘图
    history = sim.get_state_history()
    print(f"\n状态历史形状: {history.shape}")
    print(f"迭代次数（包括初始状态）: {len(history)}")
    
    plt.figure(figsize=(12, 6))
    
    # 绘制所有智能体的状态演化
    for agent_id in range(sim.n_agents):
        plt.plot(history[:, agent_id], alpha=0.7, linewidth=1.5, label=f'Agent {agent_id}')
    
    # 添加平均值线
    mean_values = np.mean(history, axis=1)
    plt.plot(mean_values, 'k--', linewidth=2, label='平均值')
    
    plt.xlabel('迭代轮数', fontsize=12)
    plt.ylabel('状态值', fontsize=12)
    plt.title(f'DeGroot共识 - 全连接网络 (10个智能体, {convergence_rounds}轮收敛)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(figures_dir, '01_complete_graph_deGroot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.show()
    
    # 测试其他拓扑
    test_topologies = [('ring', '环形网络'), ('star', '星型网络')]
    
    for topology_type, topology_name in test_topologies:
        print("\n" + "=" * 60)
        print(f"测试：{topology_name}，DeGroot共识")
        print("=" * 60)
        
        sim2 = ConsensusSimulator(n_agents=10, topology=topology_type)
        convergence_rounds2 = sim2.run_until_convergence(max_iterations=200, tolerance=1e-5)
        history2 = sim2.get_state_history()
        
        plt.figure(figsize=(12, 6))
        for agent_id in range(sim2.n_agents):
            plt.plot(history2[:, agent_id], alpha=0.7, linewidth=1.5)
        
        mean_values2 = np.mean(history2, axis=1)
        plt.plot(mean_values2, 'k--', linewidth=2, label='平均值')
        
        plt.xlabel('迭代轮数', fontsize=12)
        plt.ylabel('状态值', fontsize=12)
        plt.title(f'DeGroot共识 - {topology_name} (10个智能体, {convergence_rounds2}轮收敛)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(figures_dir, f'02_{topology_type}_graph_deGroot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
        plt.show()

if __name__ == '__main__':
    test_basic_simulation()