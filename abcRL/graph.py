import re
import os
import dgl
import torch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import bisect
import random
from dgl.nn.pytorch import GraphConv


def parse_netlist(netlist):
    inputs = re.findall(r'input\s+([\w\s,]+);', netlist)
    outputs = re.findall(r'output\s+([\w\s,]+);', netlist)
    wires = re.findall(r'wire\s+([\w\s,]+);', netlist)
    
    gates = re.findall(r'(\w+)\s+\w+\s*\(\s*([\w\s,]+)\s*\);', netlist)
    
    inputs = [i.strip() for i in ','.join(inputs).split(',')]
    outputs = [o.strip() for o in ','.join(outputs).split(',')]
    wires = [w.strip() for w in ','.join(wires).split(',')]
    
    return inputs, outputs, wires, gates


def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def extract_dgl_graph(start_node, input_netlist_path):
    # input_netlist_path = os.path.join('release','netlists', 'design1.v') 
    netlist_content = read_file(input_netlist_path)
    node_pattern = r'n\d+'
    nodes = set(re.findall(node_pattern, netlist_content))
    
    node_map = {name: int(re.findall(r'\d+', name)[0]) for name in nodes}
    node_count = len(nodes)
    print(f"Total nodes: {node_count}")

    gate_map = {'or': 1, 'and': 2, 'not': 3, 'xor': 4, 'xnor': 5, 'nor': 6}
    features = torch.zeros(node_count, len(gate_map))
    g = dgl.graph(([], []))
    g.add_nodes(node_count)

    assignments = parse_netlist(netlist_content)
    for gate in assignments[3]:
        gate_type, signal_list = gate
        if gate_type not in gate_map: 
            continue
        signals = [s.strip() for s in signal_list.split(',')]
        output = signals.pop(0) 
        input1 = signals.pop(0)
        output_idx = node_map[output]
        input1_idx = node_map[input1]
        g.add_edges(input1_idx, output_idx)
        print(gate_type, input1_idx, output_idx)
        features[output_idx][gate_map[gate_type] - 1] = 1.0
        if(len(signals)):
            input2 = signals.pop(0)
            input2_idx = node_map[input2]
            g.add_edges(input2_idx, output_idx)

    g.ndata['feat'] = features
    bidirected_G = dgl.to_bidirected(g)
    subgraph, _ = dgl.khop_in_subgraph(bidirected_G, [73], 1)
    # print("\nSubgraph Nodes:", subgraph.ndata[dgl.NID].tolist())
    # print("Subgraph Edges:", subgraph.edges(), g.edges())
    return subgraph
