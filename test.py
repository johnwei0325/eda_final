import numpy as np
# import graph as G
import torch
from dgl.nn.pytorch import GraphConv
import dgl
from gym import spaces
import os
import re
import subprocess
import json
import train_transformer as T
import gate_level as Gate_Level
# Define the command to run in the Docker container
# docker_command = [
#     "docker", "run", "-it",
#      "--rm", 
#     "-v", f"{os.getcwd()}:/workspace",  # Mount the current directory to /workspace in the container
#     "my-cost-estimator", 
#     "bash", "-c", 
#     "./cost_estimators/cost_estimator_2 -library /lib/lib1.json -netlist /examples/toy_case1.v -output cf2_ex1.out"
# ]

base_path = os.getcwd()
cost_estimators_path = os.path.join(base_path, "release/cost_estimators")
lib_path = os.path.join(base_path, "release/lib")
examples_path = os.path.join(base_path, "release/examples")
netlists_path = os.path.join(base_path, "release/netlists")
print(cost_estimators_path)
# Docker command to run the container with volume mounts
# docker_command = [
#     "docker", "run","--platform", "linux/amd64", "-it", "--rm",
#     "-v", f"{cost_estimators_path}:/workspace/cost_estimators",
#     "-v", f"{lib_path}:/workspace//lib",
#     "-v", f"{examples_path}:/workspace//examples",
#     "-v", f"{netlists_path}:/workspace//netlists",
#     "my-cost-estimator",
#     "bash", "-c",
#     "./../cost_estimator_2 -library /../lib/lib1.json -netlist /../examples/toy_case1.v -output cf2_ex1.out"
# ]

def generate_verilog_module(module_name, inputs, outputs, assigns):
    inputs_str = ', '.join(inputs)
    outputs_str = ', '.join(outputs)
    wires_str = ', '.join(wires)
    assigns_str = '\n'.join(assigns)
    
    verilog_module = f"""
module ({inputs_str}, {outputs_str});
input {inputs_str};
output {outputs_str};
wire {wires_str};
{assigns_str}
endmodule
"""
    return verilog_module

def parse_netlist(netlist):
    inputs = re.findall(r'input\s+([\w\s,]+);', netlist)
    outputs = re.findall(r'output\s+([\w\s,]+);', netlist)
    wires = re.findall(r'wire\s+([\w\s,]+);', netlist)
    
    # gates = re.findall(r'(\w+)\s+\w+\s*\(\s*([\w\s,]+)\s*\);', netlist)
    gates = netlist
    
    inputs = [i.strip() for i in ','.join(inputs).split(',')]
    outputs = [o.strip() for o in ','.join(outputs).split(',')]
    wires = [w.strip() for w in ','.join(wires).split(',')]
    
    return inputs, outputs, wires, gates

def reorder_parameters(match):
    if match.group(3):  # Three variables case
        return f"( {match.group(2).strip()} , {match.group(3).strip()} , {match.group(1).strip()} )"
    else:  # Two variables case
        return f"( {match.group(2).strip()} , {match.group(1).strip()} )"

def transform_verilog(input_file, output_file, sample):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    transformed_lines = []
    pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)(?:\s*,\s*([^,]+))?\s*\)'
    i = 0
    for line in lines:
        # Check if the line contains a gate definition
        if line.strip().startswith(('or ', 'xnor ', 'xor ', 'nor ', 'not ', 'and ', 'nand ', 'buf ', 'not_1 ', 'nand_1 ')):
            parts = line.split()
            # Find the gate name and append _1
            if len(parts) > 1 and parts[1].startswith('g'):
                gate_name = parts[0]
                gate_version = sample[i]
                gate_name_updated = gate_name.rstrip(';') + '_' + str(gate_version+1)
                # gate_name_updated = gate_name.rstrip(';') + '_1' 
                i+=1
                parts[0] = gate_name_updated
                transformed_line = ' '.join(parts) + '\n'
                transformed_line = re.sub(pattern, reorder_parameters, transformed_line)
                transformed_lines.append(' '+transformed_line)
            else:
                transformed_lines.append(line)
        else:
            transformed_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(transformed_lines)
    return "File transformation complete."


if __name__ == "__main__":
    cost_function = 3
    netlist_version = 3
    netlist_number = 1

    verilogfile = os.path.join('netlists', f'design{netlist_number}-{netlist_version}.v')
    output_netlist_path = os.path.join('examples', 'toy_case3.v')  
    # result = transform_verilog(verilogfile, output_netlist_path)
    # start_node = 50
    # graph = G.extract_dgl_graph(start_node, verilogfile)
    # with open(verilogfile, 'r') as file:
    #     lines = file.readlines()
    # print(graph.ndata['feat'])
    # with open('best_output/output1.json', 'r') as file:
    #     predict = json.load(file)
    # result = transform_verilog(verilogfile, output_netlist_path, predict)
    # print(result)

    num_gates = T.get_gate_count(verilogfile)  # Example number of gates
    num_versions = 8 if netlist_version == 2 else 6  # Example number of gate versions
    d_model = 64
    nhead = 4
    num_layers = 3

    # Model instantiation
    gate_level = Gate_Level.get_level(verilogfile)
    print(gate_level)
    # model = T.TransformerModel(num_gates, num_versions, d_model, nhead, num_layers)
    top_level = max(gate_level)+1
    # model = T.TransformerModel2(len(gate_level), num_versions, top_level, d_model, nhead, num_layers)
    model = T.TransformerModel1(len(gate_level), num_versions, d_model, nhead, num_layers)
    command = [
        f"./cost_estimators/cost_estimator_{cost_function}",
        "-library", "lib/lib1.json",
        "-netlist", "examples/toy_case3.v",
        "-output", "cf2_ex1.out"
        ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    pattern = r"[-+]?\d*\.\d+|\d+"
    with open(f'best_output/output{netlist_number}-{netlist_version}_{cost_function}.json', 'r') as file:
        predict = json.load(file)
    # predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 2]
    # predict = [2, 2, 2, 2, 2, 3, 3, 2, 5, 2, 2, 3, 2, 2, 2, 4, 2, 3, 2, 2, 3, 3, 5, 2, 2, 2, 5, 3, 5, 2, 3, 3, 2, 2, 2, 5, 2, 3, 2, 5, 3, 2, 2, 2, 5, 5, 2, 2, 5, 3, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 3, 2, 2, 5, 2, 3, 5, 3, 2, 3, 3, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 5, 2, 5, 2, 2, 3, 3, 2, 2, 3, 3, 3, 5, 5, 2, 2, 2, 3, 2, 2, 5, 4, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 0, 5, 2, 2, 5, 3, 5, 3, 3, 2, 5, 2, 3, 3, 2, 3, 2, 5, 5, 2, 2, 0, 2, 2, 3, 5, 2, 2, 3, 2, 2, 2, 4, 3, 2, 2, 2, 5, 5, 2, 3, 5, 3, 3, 2, 2, 2, 2, 2, 2, 3, 5, 4, 2, 5, 2, 2, 3, 5, 3, 2, 2, 5, 2, 2, 2, 2, 2, 3, 3, 2, 2, 5, 3, 2, 2, 3, 5, 2, 2, 5, 2, 2, 2, 3, 3, 5, 3, 5, 3, 3, 2, 2, 5, 3, 5, 3, 2, 0, 3, 3, 2, 2, 5, 3, 0, 2, 2, 3, 5, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 5, 5, 0, 2, 2, 3, 2, 3, 3, 4, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 5, 2, 5, 2, 3, 2, 5, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 5, 2, 2, 2, 3, 2, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 3, 5, 2, 2, 5, 3, 2, 5, 2, 3, 2, 2, 2, 4, 2, 2, 3, 3, 3, 5, 3, 5, 2, 2, 3, 2, 3, 0, 2, 2, 2, 2, 4, 2, 2, 2, 5, 2, 3, 5, 2, 2, 5, 2, 2, 2, 5, 3, 3, 2, 2, 0, 2, 2, 5, 3, 2, 3, 3, 2, 3, 2, 5, 3, 2, 3, 2, 2, 2, 5, 2, 2, 3, 3, 5, 3, 5, 5, 2, 2, 5, 3, 2, 0, 2, 2, 5, 2, 3, 3, 5, 2, 3, 5, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 3, 3, 2, 3, 2, 2, 2, 3, 5, 2, 4, 2, 3, 3, 5, 2, 2, 3, 3, 2, 2, 2, 5, 5, 5, 2, 2, 2, 2, 3, 5, 2, 2, 5, 2, 5, 3, 5, 5, 2, 2, 3, 5, 2, 5, 2, 5, 2, 3, 3, 3, 5, 2, 2, 2, 3, 3, 5, 2, 3, 2, 2, 3, 2, 3, 2, 5, 5, 3, 2, 5, 2, 3, 5, 3, 2, 3, 3, 3, 0, 5, 3, 2, 5, 3, 2, 3, 2, 3, 2, 2, 2, 2, 5, 2, 5, 2, 3, 2, 3, 2, 2, 2, 2, 3, 2, 3, 2, 2, 3, 5, 2, 2, 3, 5, 5, 3, 5, 3, 2, 2, 3, 2, 5, 5, 3, 3, 2, 2, 5, 3, 2, 3, 2, 2, 2, 2, 2, 2, 3, 5, 2, 5, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 0, 3, 3, 0, 2, 3, 3, 2, 2, 2, 2, 4, 2, 2, 5, 2, 2, 3, 2, 2, 2, 2, 5, 5, 2, 3, 3, 2, 3, 3, 3, 3, 5, 2, 2, 5, 2, 2, 3, 5, 2, 2, 2, 2, 2, 3, 3, 2, 3, 5, 5, 3, 2, 3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 5, 2, 2, 2, 2, 5, 3, 3, 5, 5, 5, 3, 2, 2, 2, 2, 2, 4, 2, 2, 3, 3, 2, 3, 2, 2, 3, 3, 5, 5, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 2, 2, 2, 2, 5, 3, 3, 2, 5, 0, 5, 2, 3, 5, 3, 2, 2, 2, 5, 2, 3, 3, 3, 2, 5, 2, 3, 2, 2, 5, 3, 2, 2, 3, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 3, 2, 5, 2, 2, 3, 5, 3, 3, 2, 5, 2, 2, 2, 2, 5, 3, 5, 3, 2, 2, 2, 3, 5, 5, 3, 3, 2, 2, 3, 2, 2, 2, 5, 5, 2, 3, 5, 3, 3, 3, 3, 2, 3, 3, 5, 2, 2, 2, 2, 5, 2, 5, 2, 4, 2, 2, 2, 0, 3, 5, 2, 5, 3, 2, 2, 2, 5, 3, 2, 3, 2, 2, 3, 3, 3, 2, 5, 2, 3, 3, 3, 5, 3, 3, 2, 3, 2, 2, 2, 2, 5, 5, 2, 2, 2, 5, 2, 2, 2, 2, 5, 3, 2, 2, 3, 2, 3, 3, 2, 2, 5, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 5, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 4, 2, 2, 2, 5, 2, 2, 2, 3, 3, 5, 3, 2, 5, 5, 3, 2, 3, 2, 3, 2, 2, 3, 3, 0, 2, 2, 5, 5, 2, 2, 3, 2, 3, 5, 0, 5, 2, 2, 2, 3, 2, 2, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 5, 3, 3, 5, 5, 2, 3, 2, 2, 3, 2, 2, 3, 5, 2, 3, 2, 3, 2, 2, 3, 2, 5, 2, 3, 3, 4, 2, 2, 4, 3, 2, 2, 5, 5, 2, 5, 2, 2, 5, 3, 2, 5, 3, 5, 3, 2, 5, 2, 5, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 2, 3, 5, 2, 5, 2, 2, 2, 5, 2, 2, 2, 4, 2, 3, 3, 2, 4, 5, 2, 2, 0, 2, 2, 3, 2, 2, 3, 2, 5, 2, 2, 2, 2, 2, 2, 2, 3, 3, 5, 2, 3, 2, 2, 2, 2, 2, 5, 5, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 3, 3, 2, 3, 3, 5, 2, 3, 2, 3, 2, 3, 2, 5, 0, 2, 3, 2, 2, 3, 2, 4, 3, 2, 2, 2, 3, 4, 2, 3, 2, 2, 3, 3, 2, 3, 2, 5, 5, 3, 2, 3, 2, 3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 3, 2, 3, 5, 5, 2, 2, 2, 3, 2, 3, 3, 5, 2, 2, 2, 2, 5, 3, 2, 5, 2, 2, 2, 2, 2, 5, 2, 2, 3, 2, 3, 3, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 5, 3, 2, 2, 2, 5, 2, 0, 3, 3, 2, 5, 2, 2, 5, 5, 5, 2, 5, 2, 2, 2, 2, 3, 4, 2, 5, 2, 4, 5, 3, 2, 2, 2, 2, 5, 2, 2, 3, 2, 5, 3, 2, 5, 3, 2, 2, 2, 3, 3, 2, 2, 3, 2, 4, 2, 3, 2, 3, 2, 2, 3, 2, 5, 3, 3, 2, 3, 2, 2, 2, 2, 4, 5, 2, 3, 2, 2, 2, 3, 2, 3, 4, 3, 2, 2, 2, 3, 2, 3, 2, 2, 5, 3, 3, 5, 2, 3, 3, 2, 2, 3, 2, 5, 3, 3, 5, 2, 2, 5, 3, 2, 2, 5, 2, 3, 3, 5, 2, 2, 3, 5, 5, 2, 3, 3, 5, 3, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 3, 3, 2, 3, 3, 2, 2, 3, 2, 5, 3, 3, 2, 2, 4, 2, 2, 3, 2, 3, 3, 3, 2, 2, 3, 4, 3, 3, 2, 2, 5, 2, 2, 3, 2, 5, 4, 2, 3, 3, 2, 2, 2, 5, 5, 3, 2, 5, 3, 2, 2, 2, 2, 2, 0, 3, 3, 5, 3, 3, 5, 3, 2, 3, 2, 2, 2, 2, 2, 2, 5, 2, 2, 2, 3, 2, 2, 5, 0, 2, 4, 0, 3, 3, 2, 3, 5, 2, 2, 2, 3, 3, 5, 3, 3, 5, 2, 3, 2, 2, 5, 5, 2, 2, 2, 2, 2, 5, 2, 5, 2, 2, 2, 2, 2, 2, 3, 3, 5, 3, 3, 2, 5, 2, 3, 2, 5, 2, 2, 2, 2, 2, 2, 0, 5, 5, 5, 5, 5, 5, 2, 2, 5, 2, 2, 3, 2, 2, 2, 5, 2, 5, 2, 2, 2, 2, 5, 2, 2, 5, 2, 3, 2, 3, 3, 4, 3, 2, 2, 2, 2, 5, 2, 2, 2, 3, 2, 3, 5, 2, 2, 3, 0, 2, 2, 2, 2, 3, 3, 2, 5, 3, 2, 2, 2, 2, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 5, 2, 2, 5, 2, 5, 3, 3, 2, 5, 2, 5, 3, 5, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 2, 2, 5, 3, 3, 2, 2, 2, 5, 2, 5, 5, 3, 2, 5, 2, 3, 3, 2, 2, 2, 2, 5, 2, 3, 2, 2, 5, 2, 2, 2, 5, 2, 3, 2, 3, 3, 2, 5, 2, 2, 2, 5, 3, 2, 3, 2, 2, 3, 5, 3, 4, 5, 3, 2, 3, 0, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 2, 5, 2, 2, 2, 5, 3, 2, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 3, 3, 2, 3, 5, 2, 3, 3, 2, 2, 2, 2, 5, 2, 5, 2, 0, 3, 3, 3, 5, 3, 3, 5, 2, 2, 2, 2, 2, 3, 5, 0, 3, 3, 3, 2, 2, 2, 3, 5, 2, 2, 0, 2, 2, 3, 2, 2, 2, 2, 2, 5, 2, 2, 2, 5, 2, 3, 2, 3, 5, 5, 3, 3, 3, 2, 4, 5, 5]
    print(len(predict))
    result = transform_verilog(verilogfile, output_netlist_path, predict)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    matches = re.findall(pattern, result.stdout)
    best = matches[0]
    print('g' ,result.stdout, result.stderr)
    state_dict = torch.load(f'model/best_model{netlist_number}-{netlist_version}_{cost_function}.pth')

    # # Load the state dictionary into your model
    model.load_state_dict(state_dict)
    model.eval()
    random_samples = T.random_sampling(20, num_gates, num_versions, gate_level)
    data_costs = []
    for sample in random_samples:
        result = transform_verilog(verilogfile, output_netlist_path, sample)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        matches = re.findall(pattern, result.stdout)
        data_costs.append(float(matches[0]))
    # data = []
    # for circuit in random_samples:
    #     for idx in range(len(circuit)):
    #         # print(len(circuit), len(gate_level))
    #         circuit[idx] = [circuit[idx], gate_level[idx]]
    #     data.append(circuit)

    # random_samples = data  
    
        
    dataset = T.CircuitDataset(random_samples, data_costs)
    dataloader = T.DataLoader(dataset, batch_size=1, shuffle=True)

    compare = []
    cost = []
    cost2 = []
    for epoch in range(1):
        for inputs, target_cost in dataloader:
            outputs = model(inputs)
            predicted_versions = outputs.argmax(dim=-1)
            estimated_costs = []
            if isinstance(predicted_versions.tolist()[0],int):
                result = transform_verilog(verilogfile, output_netlist_path, predicted_versions.tolist())
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                matches = re.findall(pattern, result.stdout)
                estimated_costs.append(float(matches[0]))
            else:    
                for predict in predicted_versions.tolist():
                    # print(predicted_versions.tolist(), predict)
                    result = transform_verilog(verilogfile, output_netlist_path, predict)
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    matches = re.findall(pattern, result.stdout)
                    estimated_costs.append(float(matches[0]))

            estimated_cost_tensor = torch.tensor([estimated_costs], dtype=torch.float, requires_grad=True)
            cost.append(estimated_cost_tensor.mean())
            cost2.append(target_cost.mean())
            compare.append(estimated_cost_tensor - target_cost)
            print(estimated_cost_tensor ,target_cost)

    compare_tensor =torch.tensor(compare, dtype=torch.float)
    mean = torch.tensor([cost2], dtype=torch.float, requires_grad=True).mean()
    print(torch.tensor(float(best)), mean)
    print("Compare:", torch.tensor(float(best))/mean)

