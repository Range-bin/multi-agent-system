# network_generator.py
import networkx as nx
import numpy as np
import random

def generate_topology(topology_type, n_agents, **kwargs):
    """
    根据类型生成网络拓扑。
    参数:
        topology_type: 'complete'（全连接）, 'ring'（环形）, 'star'（星型）, 'small_world'（小世界）
        n_agents: 智能体数量
        **kwargs: 其他参数
    """
    if topology_type == 'complete':
        G = nx.complete_graph(n_agents)
        
    elif topology_type == 'ring':
        # 默认每个节点与左右各1个邻居相连（k=2）
        G = nx.cycle_graph(n_agents)

    elif topology_type == 'star':
        # 星型网络
        G = nx.star_graph(n_agents - 1)
        # 重命名节点，使其从0开始
        mapping = {i: i-1 for i in range(1, n_agents)}
        mapping[0] = n_agents - 1  # 中心节点
        G = nx.relabel_nodes(G, mapping)

    elif topology_type == 'small_world':
        # 小世界网络（Watts-Strogatz模型）
        k = kwargs.get('k', 4)    # 每个节点连接的邻居数（偶数）
        p = kwargs.get('p', 0.1)  # 重连概率
        G = nx.watts_strogatz_graph(n_agents, k, p)
        
    else:
        raise ValueError(f"未知的拓扑类型: {topology_type}")
    
    return G

def get_adjacency_list(G):
    """将networkx图转换为邻接列表（字典）"""
    adj_list = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        adj_list[node] = neighbors
        # 调试输出
        if node < 3:  # 只显示前3个节点
            print(f"  节点{node}的邻居: {neighbors}")
    return adj_list