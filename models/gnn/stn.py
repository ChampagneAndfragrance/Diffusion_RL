import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
# from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.recurrent.attentiontemporalgcn import A3TGCN2
# from torch_geometric_temporal.nn.recurrent.attention.temporalgcn import A3TGCN
from itertools import combinations
from torch_geometric.nn.glob.glob import global_mean_pool

def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)

class STGCNEncoder(nn.Module):
    def __init__(self, input_dim, gnn_hidden, periods, batch_size):
        super(STGCNEncoder, self).__init__()
        self.gnn_hidden = gnn_hidden
        self.batch_size = batch_size
        self.gnn = A3TGCN2(in_channels=input_dim,
                           out_channels=gnn_hidden,
                           periods=periods,
                           batch_size=batch_size,
                           )
        self.initialized_graphs = {n: self.initialize_graph(n) for n in range(83, 95)}

        # self.global_mean_pool = global_mean_pool()
        
    def initialize_graph(self, num_agents):
        # initialize graph just from 
        s, e = fully_connected(num_agents)
        s = s.to(self.device)
        e = e.to(self.device)
        edge_indices = torch.stack([s, e], dim=0)
        return edge_indices

    @property
    def device(self):
        return next(self.parameters()).device
        # return "cuda"

    def create_batched_graph(self):
        pass

    def forward(self, x):
        # x is (0: batch_size, 1: seq_len, 2: num_agents, 3: features)
        # input_tensor = torch.randn((batch_size, seq_len, num_agents, features))        
        agent_obs, hideout_obs, timestep_obs, num_agents = x
        agent_obs = agent_obs.to(self.device).float()

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        # num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]
        
        agent_obs = agent_obs.permute(0, 2, 3, 1) # (batch_size, num_agents, features, seq_len)

        data_list = []
        # for n, batch in zip(num_agents, list(range(batch_size))):
        #     n = int(n.item())
        #     edge_index = self.initialized_graphs[n]
        #     data = Data(edge_index=edge_index)
        #     data_list.append(data)

        # batch = Batch()
        # batched_data = batch.from_data_list(data_list)

        edge_index = self.initialized_graphs[83]

        x = self.gnn(agent_obs, edge_index)
        # returns (batch_size, num_agents (nodes), hidden_dim)
        # x = global_mean_pool(x)
        x = x.mean(dim=1)
        res = torch.cat((x, hideout_obs, timestep_obs), dim=-1)
        # print(conv1(batched_data.x, batched_data.edge_index).shape)

        # location_lstm_output = self.location_lstm(location_obs)
        # res = torch.cat((res, hideout_obs, timestep_obs), dim=-1)
        # return self.linear(res)

        return res