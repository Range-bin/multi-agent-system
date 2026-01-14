# experiments/fix_test.py
import sys
import os
import numpy as np
import networkx as nx

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.network_generator import generate_topology, get_adjacency_list

print("=" * 60)
print("测试网络生成器")
print("=" * 60)

# 测试1：环形网络
print("\n1. 测试环形网络 (n=5)")
G_ring = generate_topology('ring', 5)
print(f"   节点: {list(G_ring.nodes())}")
print(f"   边: {list(G_ring.edges())}")

adj_ring = get_adjacency_list(G_ring)
print(f"   邻接列表: {adj_ring}")

# 测试2：全连接网络
print("\n2. 测试全连接网络 (n=5)")
G_complete = generate_topology('complete', 5)
adj_complete = get_adjacency_list(G_complete)
print(f"   邻接列表: {adj_complete}")

# 测试3：星型网络
print("\n3. 测试星型网络 (n=5)")
G_star = generate_topology('star', 5)
adj_star = get_adjacency_list(G_star)
print(f"   邻接列表: {adj_star}")

# 测试4：小世界网络
print("\n4. 测试小世界网络 (n=5, k=2, p=0.1)")
G_sw = generate_topology('small_world', 5, k=2, p=0.1)
adj_sw = get_adjacency_list(G_sw)
print(f"   邻接列表: {adj_sw}")