import re
from collections import defaultdict


def calculate_level(signal, levels, dependencies):
    if signal in levels:
        return levels[signal]
    if signal not in dependencies:
        return None
    max_level = 0
    for dep in dependencies[signal]:
        dep_level = calculate_level(dep, levels, dependencies)
        if dep_level is None:
            continue
        max_level = max(max_level, dep_level)
    levels[signal] = max_level + 1
    return levels[signal]

def get_level(verilogfile):
    # Path to the Verilog file
    # verilogfile = 'release/netlists/design1.v'

    # Read the Verilog file
    with open(verilogfile, 'r') as file:
        lines = file.readlines()

    # Convert lines to a single string for easy parsing
    netlist = ''.join(lines)

    # Define primary inputs and outputs
    primary_inputs = set(re.findall(r'input\s+([\w\s,]+);', netlist))
    primary_inputs = set(re.findall(r'\b\w+\b', ','.join(primary_inputs)))
    primary_outputs = set(re.findall(r'output\s+([\w\s,]+);', netlist))
    primary_outputs = set(re.findall(r'\b\w+\b', ','.join(primary_outputs)))

    # Define wires and gates
    wires = set(re.findall(r'wire\s+([\w\s,]+);', netlist))
    wires = set(re.findall(r'\b\w+\b', ','.join(wires)))
    gates = re.findall(r'(\w+)\s+g\d+\s*\(([^)]+)\);', netlist)

    # Initialize the level of primary inputs to 0
    levels = {pin: 0 for pin in primary_inputs}

    # Initialize the dependencies for each wire
    dependencies = defaultdict(list)
    for gate_type, connections in gates:
        connections = connections.replace(' ', '').split(',')
        output = connections[0]
        inputs = connections[1:]
        for inp in inputs:
            dependencies[output].append(inp)

    # Function to calculate the level of a gate


    # Calculate levels for all gates
    for gate_type, connections in gates:
        connections = connections.replace(' ', '').split(',')
        output = connections[0]
        calculate_level(output, levels, dependencies)

    # Print the levels
    gate_levels = []
    for gate_type, connections in gates:
        connections = connections.replace(' ', '').split(',')
        output = connections[0]
        level = levels.get(output, 'Unknown')
        # print(f'Gate {output} (Type: {gate_type}) is at level {level}')
        gate_levels.append(level)
    return gate_levels
