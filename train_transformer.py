import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import test as T
# import graph as G
import os
import subprocess
import re 
import dgl
import json
import math
import gate_level as Gate_Level

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()
    
def random_sampling(num_samples, num_gates, num_versions, gate_level):
    data = []
    circuits = [np.random.randint(num_versions, size=len(gate_level)).tolist() for _ in range(num_samples)]
    
    return circuits

def get_gate_count(input_netlist_path):
    netlist_content = read_file(input_netlist_path)
    node_pattern = r'n\d+'
    nodes = set(re.findall(node_pattern, netlist_content))
    
    node_map = {name: int(re.findall(r'\d+', name)[0]) for name in nodes}
    node_count = len(nodes)
    print(f"Total nodes: {node_count}")
    return node_count

def get_last_var_and_number(input_string):
    cleaned_string = input_string.strip('()')
    variables = cleaned_string.split(',')
    last_var = variables[-1].strip()
    match = re.search(r'\d+', last_var)
    if match:
        number = int(match.group())
    else:
        number = None
    return number

def transform_verilog(input_file, gate_level):
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
                print(parts[1])
                gate_level[get_last_var_and_number(parts[2])] += 1

class CircuitDataset(Dataset):
    def __init__(self, circuits, costs):
        self.circuits = circuits
        self.costs = costs

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, idx):
        return torch.tensor(self.circuits[idx], dtype=torch.long), torch.tensor(self.costs[idx], dtype=torch.float)

    def to_dict(self):
        # Convert the object to a dictionary
        return {"data": self.circuits, "cost": self.costs}
    

class TransformerModel1(nn.Module):
    def __init__(self, num_gates, num_versions, d_model=64, nhead=4, num_layers=3):
        super(TransformerModel1, self).__init__()
        self.embedding = nn.Embedding(num_versions, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, num_versions)
        self.d_model = d_model
    def forward(self, src):
        # src shape: [batch_size, num_gates]
        # Here we assume batch_size = 1 for a single circuit
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        embedded = embedded.transpose(0, 1)  # transformer expects [seq_len, batch_size, d_model]
        
        # Ensure batch_size = 1, i.e., single circuit
        transformer_out = self.transformer(embedded, embedded)
        transformer_out = transformer_out.transpose(0, 1)  # back to [batch_size, num_gates, d_model]
        
        output = self.fc_out(transformer_out)
        return output.squeeze(0)
    
class TransformerModel2(nn.Module):
    def __init__(self, num_gates, num_versions, top_level, d_model=64, nhead=4, num_layers=3):
        super(TransformerModel2, self).__init__()
        self.num_gates = num_gates
        self.num_versions = num_versions
        self.top_level = top_level
        self.d_model = d_model
        
        # Embedding layers for each part of the [a, b] pair
        self.embedding_a = nn.Embedding(num_versions, d_model // 2)
        self.embedding_b = nn.Embedding(top_level, d_model // 2)
        
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, num_versions)

    def forward(self, src):
        # src shape: [batch_size, num_gates, 2]
        batch_size = src.size(0)
        num_gates = src.size(1)
        
        # Separate a and b components
        a_part = src[:, :, 0]  # Shape: [batch_size, num_gates]
        b_part = src[:, :, 1]  # Shape: [batch_size, num_gates]
        # print("b_part values:", b_part)
        # print("Valid range: 0 to", self.top_level - 1)
        # Get embeddings
        embedded_a = self.embedding_a(a_part)  # Shape: [batch_size, num_gates, d_model//2]
        embedded_b = self.embedding_b(b_part)  # Shape: [batch_size, num_gates, d_model//2]

        # Concatenate embeddings
        embedded = torch.cat((embedded_a, embedded_b), dim=2)  # Shape: [batch_size, num_gates, d_model]

        # Scale and transpose for transformer
        embedded = embedded * math.sqrt(self.d_model)
        embedded = embedded.transpose(0, 1)  # Transformer expects [seq_len, batch_size, d_model]

        # Pass through the transformer
        transformer_out = self.transformer(embedded, embedded)
        transformer_out = transformer_out.transpose(0, 1)  # Back to [batch_size, num_gates, d_model]

        # Apply the final linear layer
        output = self.fc_out(transformer_out)  # Shape: [batch_size, num_gates, num_versions]
        
        return output
    
# Hyperparameters
def get_dataset(num_gates, num_versions, verilogfile, output_netlist_path, dataset_path, cost_function, gate_level):
    random_samples = random_sampling(5000, num_gates, num_versions, gate_level)
    # print(random_samples)
    command = [
        f"./cost_estimators/cost_estimator_{cost_function}",
        "-library", "lib/lib1.json",
        "-netlist", "examples/toy_case3.v",
        "-output", "cf2_ex1.out"
        ]
    data_costs = []
    pattern = r"[-+]?\d*\.\d+|\d+"
    for sample in random_samples:
        result = T.transform_verilog(verilogfile, output_netlist_path, sample)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        matches = re.findall(pattern, result.stdout)
        # print(result.stdout, result.stderr)
        data_costs.append(float(matches[0]))

    data = []
    # for circuit in random_samples:
    #     circuit2 = circuit
    #     for idx in range(len(circuit)):
    #         circuit2[idx] = [circuit[idx], gate_level[idx]]
    #     data.append(circuit2)

    # dataset1 = CircuitDataset(random_samples, data_costs)
    # random_samples = data  
    dataset = CircuitDataset(random_samples, data_costs)
    with open(dataset_path, 'w') as json_file:
        json.dump(dataset.to_dict(), json_file)
    return dataset

def load_dataset(datapath):
    with open(datapath, 'r') as file:
    # Step 3: Load the JSON data
        data_dict = json.load(file)
    # print(min(data_dict['cost']))
    dataset = CircuitDataset(data_dict["data"][:1000], data_dict["cost"][:1000])
    return dataset

if __name__ == "__main__":  
    cost_function = 3
    netlist_version = 3
    netlist_number = 1
    index_of_file = str(netlist_number) + "-" + str(netlist_version) + "_" + str(cost_function)
    print(f'best_output/output{netlist_number}-{netlist_version}_{cost_function}.json')
    

    verilogfile = os.path.join('netlists', f'design{netlist_number}-{netlist_version}.v')
    output_netlist_path = os.path.join('examples', 'toy_case3.v')
    num_gates = get_gate_count(verilogfile)
    gate_level = Gate_Level.get_level(verilogfile)
    # print(gate_level)
    
    # g = G.extract_dgl_graph(verilogfile)
    # start_node = 1
    # subg = G.extract_subgraph(start_node, g, 1)
    # print(subg)
    command = [
        f"./cost_estimators/cost_estimator_{cost_function}",
        "-library", "lib/lib1.json",
        "-netlist", "examples/toy_case3.v",
        "-output", "cf2_ex1.out"
        ]
    pattern = r"[-+]?\d*\.\d+|\d+"
      # Example number of gates
    num_versions = 8 if netlist_version == 2 else 6  # Example number of gate versions
    print(num_versions)
    d_model = 64
    nhead = 4
    num_layers = 3
    # Model instantiation
    top_level = max(gate_level)+1
    # model = TransformerModel2(len(gate_level), num_versions, top_level, d_model, nhead, num_layers)
    model = TransformerModel1(len(gate_level), num_versions, d_model, nhead, num_layers)
    for param in model.parameters():
        param.requires_grad = True

    # dataset = get_dataset(num_gates, num_versions, verilogfile, output_netlist_path, f'dataset/dataset{netlist_number}-{netlist_version}_{cost_function}.json', cost_function, gate_level)
    dataset = load_dataset(f'dataset/dataset{netlist_number}-{netlist_version}_{cost_function}.json')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    subset_size = 3  # Number of samples per epoch
    subset_indices = np.random.choice(indices, size=subset_size, replace=False)
    sampler = SubsetRandomSampler(subset_indices)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    # # Training setup
    criterion = nn.MSELoss()  # Mean Squared Error for simplicity
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    eval_interval = 2
    min_cost = 100000000
    min_circuit = []
    min_loss = 100000000
    with open(f'dataset/dataset{netlist_number}-{netlist_version}_{cost_function}.json', 'r') as file:
        data_dict = json.load(file)
    for epoch in range(num_epochs):
        subset_indices = np.random.choice(indices, size=subset_size, replace=False)
        sampler = SubsetRandomSampler(subset_indices)
        dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)
        for inputs, target_cost in dataloader:
            # print(len(inputs[0]), target_cost)
        #===================
        # for i in range(0,2000):
            
        #     # print(inputs, target_cost)
        #     if len(min_circuit) >0:
        #         inputs = min_circuit
        #         target_cost = min_cost
        #     else:
        #         inputs = data_dict['data'][0]
        #         target_cost = data_dict['cost'][0]
        #     inputs = torch.tensor(inputs)
        #     target_cost = torch.tensor(target_cost)
        #====================
            # print(inputs)

            optimizer.zero_grad()
            # print('ddd')
            outputs = model(inputs)
            predicted_versions = outputs.argmax(dim=-1)
            # print(outputs, predicted_versions.tolist())
            estimated_costs = []
            #==================
            # result = T.transform_verilog(verilogfile, output_netlist_path, outputs)
            # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # matches = re.findall(pattern, result.stdout)
            # estimated_costs.append(float(matches[0]))
            # temp = min(estimated_costs)
            # if temp < min_cost:    
            #     min_cost_index = estimated_costs.index(min(estimated_costs))
            #     min_circuit = predicted_versions.tolist()[min_cost_index]
            #     if isinstance(predicted_versions.tolist()[0],int):
            #         min_circuit = predicted_versions.tolist()
            #     min_cost = min(estimated_costs)
            #==================
            # print(predicted_versions.tolist())
            if isinstance(predicted_versions.tolist()[0],int):
                result = T.transform_verilog(verilogfile, output_netlist_path, predicted_versions.tolist())
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                matches = re.findall(pattern, result.stdout)
                # print("input: ", len(inputs), "output: ", len(predicted_versions.tolist()[0]), "cost: ", result.stdout, result.stderr,"target: ", target_cost)
                estimated_costs.append(float(matches[0]))
            else:    
                for predict in predicted_versions.tolist():
                    # print(predicted_versions.tolist(), predict)
                    result = T.transform_verilog(verilogfile, output_netlist_path, predict)
                    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    matches = re.findall(pattern, result.stdout)
                    estimated_costs.append(float(matches[0]))

            temp = min(estimated_costs)
            if temp < min_cost:    
                min_cost_index = estimated_costs.index(min(estimated_costs))
                min_circuit = predicted_versions.tolist()[min_cost_index]
                if isinstance(predicted_versions.tolist()[0],int):
                    min_circuit = predicted_versions.tolist()
                min_cost = min(estimated_costs)
            # print(estimated_costs, target_cost)
            # loss = criterion(torch.tensor([estimated_cost],requires_grad=True), target_cost)
            # estimated_cost = float(estimated_cost)  # Ensure `estimated_cost` is a float value
            estimated_cost_tensor = torch.tensor([estimated_costs], dtype=torch.float, requires_grad=True)
            # Calculate the loss as the difference
            loss = (estimated_cost_tensor - target_cost)/target_cost 
            loss = loss.mean()
            # print(loss)
            loss.requires_grad_(True)

            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        if loss.item() < min_loss:
            print(loss.item(), min_loss)
            min_loss = loss.item()
            best_model_state = model.state_dict()
            torch.save(best_model_state, f'model/best_model{netlist_number}-{netlist_version}_{cost_function}.pth')
            print(f'New best model saved at epoch {epoch + 1} with loss: {loss.item()}')

    print('Training completed.')

    with open(f'best_output/output{netlist_number}-{netlist_version}_{cost_function}.json', 'w') as f:
        json.dump(min_circuit, f)

    