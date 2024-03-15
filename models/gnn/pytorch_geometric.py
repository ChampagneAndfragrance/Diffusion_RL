import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import GConvGRU
# # Test on converting from lstm output to graph input
# batch_size = 64
# hidden_dim = 8
# num_agents = 60
# lstm_output = torch.randn((batch_size * num_agents, hidden_dim)).requires_grad_()

batch_size = 64
seq_len = 5
num_agents = 3
features = 2
hidden_dim = 8

input_tensor = torch.randn((batch_size, seq_len, num_agents, features))
permuted = input_tensor.permute(0, 2, 1, 3).contiguous() # (batch_size, num_agents, seq_len, features)
lstm_input = permuted.view(batch_size * num_agents, seq_len, features)

lstm_layer = nn.LSTM(input_size=features, 
                hidden_size=hidden_dim, 
                num_layers=1, 
                batch_first=True)

out, (hn, cn) = lstm_layer(lstm_input)
print(hn.shape)

reshaped = hn.view(batch_size, num_agents, hidden_dim)
print(reshaped.shape)

data_list = []
for i in range(batch_size):
    x = reshaped[i]
    edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data_list.append(data)
    # print(data.x.requires_grad)
# print(reshaped.shape)

batch = Batch()
batched_data = batch.from_data_list(data_list)

print(batched_data.x)
print(batched_data.edge_index)

conv1 = GCNConv(hidden_dim, 4)
print(conv1(batched_data.x, batched_data.edge_index).shape)

# print(batched_data)
# batched_data[0]