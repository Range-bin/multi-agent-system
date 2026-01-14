# experiments/02_test_small_world.py
"""
测试小世界网络中的共识过程
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

def test_small_world_variations():
    """测试不同参数的小世界网络"""
    figures_dir = ensure_figures_dir()
    
    # 测试不同重连概率p
    p_values = [0, 0.01, 0.1, 0.5, 1.0]
    results = []
    
    print("=" * 60)
    print("小世界网络不同重连概率对比实验")
    print("=" * 60)
    
    for p in p_values:
        print(f"\n>>> 测试: p = {p}")
        
        # 创建模拟器，传递小世界网络参数
        sim = ConsensusSimulator(
            n_agents=20,
            topology='small_world',
            k=4,  # 每个节点连接4个邻居
            p=p    # 重连概率
        )
        
        # 运行仿真
        iterations = sim.run_until_convergence(max_iterations=200, tolerance=1e-5)
        
        # 获取历史数据
        history = sim.get_state_history()
        final_std = np.std(history[-1])
        consensus_value = np.mean(history[-1])
        
        results.append({
            'p': p,
            'iterations': iterations,
            'final_std': final_std,
            'consensus_value': consensus_value
        })
        
        # 绘制状态演化图
        plt.figure(figsize=(12, 6))
        for agent_id in range(sim.n_agents):
            plt.plot(history[:, agent_id], alpha=0.6, linewidth=1)
        
        plt.xlabel('迭代轮数')
        plt.ylabel('状态值')
        plt.title(f'小世界网络共识过程 (p={p}, k=4, {iterations}轮收敛)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(figures_dir, f'small_world_p{p}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
        plt.close()  # 关闭图形，避免内存泄漏
    
    # 绘制性能对比图
    print("\n" + "=" * 60)
    print("性能对比分析")
    print("=" * 60)
    
    plt.figure(figsize=(10, 6))
    
    # 提取数据
    p_vals = [r['p'] for r in results]
    iterations = [r['iterations'] for r in results]
    
    plt.subplot(2, 1, 1)
    plt.plot(p_vals, iterations, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('重连概率 p')
    plt.ylabel('收敛所需轮数')
    plt.title('小世界网络：重连概率对收敛速度的影响')
    plt.grid(True, alpha=0.3)
    
    # 标记特殊点
    plt.annotate('环形网络 (p=0)', xy=(0, iterations[0]), 
                xytext=(0.1, iterations[0]+10),
                arrowprops=dict(arrowstyle='->'))
    plt.annotate('随机网络 (p=1)', xy=(1, iterations[-1]), 
                xytext=(0.7, iterations[-1]+10),
                arrowprops=dict(arrowstyle='->'))
    
    plt.subplot(2, 1, 2)
    # 绘制网络结构示意图（概念图）
    x_pos = [0, 0.25, 0.5, 0.75, 1.0]
    network_labels = ['环形', '近规则', '小世界', '小世界', '随机']
    
    for i, label in enumerate(network_labels):
        plt.text(x_pos[i], 0.5, label, ha='center', va='center', fontsize=12)
        plt.plot([x_pos[i], x_pos[i]], [0.3, 0.7], 'k-', alpha=0.5)
    
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('网络结构变化谱系')
    
    plt.tight_layout()
    
    # 保存对比图
    comparison_path = os.path.join(figures_dir, 'small_world_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存: {comparison_path}")
    plt.show()
    
    # 打印结果表格
    print("\n" + "-" * 60)
    print("实验结果汇总")
    print("-" * 60)
    print(f"{'p值':<8} {'收敛轮数':<12} {'最终标准差':<15} {'共识值':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['p']:<8} {r['iterations']:<12} {r['final_std']:<15.6f} {r['consensus_value']:<10.4f}")
    print("-" * 60)
    
    # 分析结论
    print("\n📊 实验结论:")
    print("1. p=0（环形网络）: 收敛最慢，信息传播路径最长")
    print("2. p=0.01~0.1（小世界网络）: 收敛速度显著提升")
    print("3. p=1（随机网络）: 收敛最快，接近全连接网络的效率")
    print("4. 小世界网络兼具高聚类系数和短平均路径长度")

def test_different_k_values():
    """测试不同邻居数k的影响"""
    print("\n" + "=" * 60)
    print("测试不同邻居数k对共识速度的影响")
    print("=" * 60)
    
    k_values = [2, 4, 6, 8]
    p_fixed = 0.1
    
    for k in k_values:
        print(f"\n>>> 测试: k = {k} (p={p_fixed})")
        
        sim = ConsensusSimulator(
            n_agents=20,
            topology='small_world',
            k=k,
            p=p_fixed
        )
        
        iterations = sim.run_until_convergence(max_iterations=150, tolerance=1e-5)
        print(f"  收敛所需轮数: {iterations}")

if __name__ == '__main__':
    test_small_world_variations()
    test_different_k_values()