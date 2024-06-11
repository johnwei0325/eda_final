import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import test as T
import os
import subprocess
import re 

def random_sampling(num_samples, num_gates, num_versions):
    return [np.random.randint(num_versions, size=num_gates).tolist() for _ in range(num_samples)]

class CircuitDataset(Dataset):
    def __init__(self, circuits, costs):
        self.circuits = circuits
        self.costs = costs

    def __len__(self):
        return len(self.circuits)

    def __getitem__(self, idx):
        return torch.tensor(self.circuits[idx], dtype=torch.long), torch.tensor(self.costs[idx], dtype=torch.float)

class TransformerModel(nn.Module):
    def __init__(self, num_gates, num_versions, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(num_versions, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, num_versions)

    def forward(self, src):
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(d_model, dtype=torch.float))
        transformer_out = self.transformer(embedded, embedded)
        output = self.fc_out(transformer_out)
        return output

# Hyperparameters
num_gates = 5000  # Example number of gates
num_versions = 5  # Example number of gate versions
d_model = 64
nhead = 4
num_layers = 3

# Model instantiation
model = TransformerModel(num_gates, num_versions, d_model, nhead, num_layers)

# Sample data (dummy example)
# circuits = [[0] * num_gates, [1] * num_gates, [2] * num_gates]  # Example circuits
# costs = [100, 80, 120]  # Corresponding costs

# Generate 1000 random samples
random_samples = random_sampling(1000, num_gates, num_versions)
# print(random_samples[0])
verilogfile = os.path.join('netlists', 'design1.v')
output_netlist_path = os.path.join('examples', 'toy_case1.v')
# command = "./cost_estimators/cost_estimator_2 -library /lib/lib1.json -netlist /examples/toy_case2.v -output cf2_ex1.out"
command = [
    "./cost_estimators/cost_estimator_2",
    "-library", "lib/lib1.json",
    "-netlist", "examples/toy_case1.v",
    "-output", "cf2_ex1.out"
    ]
data_costs = []
pattern = r"[-+]?\d*\.\d+|\d+"

for sample in random_samples:
    result = T.transform_verilog(verilogfile, output_netlist_path, sample)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    matches = re.findall(pattern, result.stdout)
    data_costs.append(matches[0])
print(data_costs)
    # print("Output:", result.stdout)
    # print("Error:", result.stderr)
# dataset = CircuitDataset(circuits, costs)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Training setup
# criterion = nn.MSELoss()  # Mean Squared Error for simplicity
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 10

# for epoch in range(num_epochs):
#     for inputs, target_cost in dataloader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         predicted_versions = outputs.argmax(dim=-1)
        
#         # Assume a function `estimate_cost` that takes predicted versions and returns the cost
#         estimated_cost = estimate_cost(predicted_versions.tolist())  # Replace with actual estimator call
#         loss = criterion(torch.tensor([estimated_cost]), target_cost)
        
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

