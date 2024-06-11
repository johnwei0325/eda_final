import numpy as np
# import graph as G
import torch
from dgl.nn.pytorch import GraphConv
import dgl
from gym import spaces
import os
import re
import subprocess

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
        if line.strip().startswith(('or ', 'xnor ', 'xor ', 'nor ', 'not ', 'and ', 'nand ', 'buf ')):
            parts = line.split()
            # Find the gate name and append _1
            if len(parts) > 1 and parts[1].startswith('g'):
                gate_name = parts[0]
                gate_name_updated = gate_name.rstrip(';') + '_' + str(sample[i]+1)
                i+=1
                parts[0] = gate_name_updated
                transformed_line = ' '.join(parts) + '\n'
                transformed_line = re.sub(pattern, reorder_parameters, transformed_line)
                transformed_lines.append(' '+transformed_line)
            else:
                transformed_lines.append(line)
        else:
            transformed_lines.append(line)

    # new_lines = []
    # for line in transformed_line:
    #     new_line = re.sub(pattern, reorder_parameters, line)
    #     new_lines.append(new_line)

    with open(output_file, 'w') as file:
        file.writelines(transformed_lines)
    return "File transformation complete."


if __name__ == "__main__":
    # verilogfile = os.path.join('netlists', 'design1.v')
    # output_netlist_path = os.path.join('examples', 'toy_case1.v')  
    # result = transform_verilog(verilogfile, output_netlist_path)
    # start_node = 50
    # graph = G.extract_dgl_graph(start_node, verilogfile)
    # with open(verilogfile, 'r') as file:
    #     lines = file.readlines()
    # print(graph.ndata['feat'])
    command = "./cost_estimators/cost_estimator_2 -library /lib/lib1.json -netlist /examples/toy_case2.v -output cf2_ex1.out"
    command = [
    "./cost_estimators/cost_estimator_2",
    "-library", "lib/lib1.json",
    "-netlist", "examples/toy_case1.v",
    "-output", "cf2_ex1.out"
    ]
    # result = subprocess.run(command, capture_output=True, text=True, check=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  

    print("Output:", result.stdout)
    print("Error:", result.stderr)

