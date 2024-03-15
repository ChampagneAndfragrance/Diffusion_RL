""" Just use the GCLSTM from pytorch geometric for the agents and concat it with the hideout information """

from torch_geometric_temporal.nn.recurrent.gc_lstm import GCLSTM
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
import torch_geometric.transforms as T
# from torch_geometric.nn import global_mean_pool
from torch_geometric.graphgym.models.pooling import global_mean_pool

from itertools import combinations

def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)

# Create graph encoder consisting of an lstm into a graph NN
class GCLSTMPrisoner(nn.Module):
    def __init__(self, input_dim, gnn_hidden, start_location=False):
        super(GCLSTMPrisoner, self).__init__()
        self.gnn_hidden = gnn_hidden
        self.start_location = start_location
        self.initialized_graphs = dict()
        self.act = nn.Tanh()
        self.gclstm = GCLSTM(in_channels=input_dim, out_channels=gnn_hidden, K=1)

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_graph(self, n_agents):
        from_nodes, to_nodes = fully_connected(n_agents)
        edges = torch.stack((from_nodes, to_nodes)).to(torch.long)
        data = Data(edge_index=edges)
        data.num_nodes = n_agents
        return data

    def forward(self, x):
        # x is (batch_size, seq_len, num_agents, features)
        # input_tensor = torch.randn((batch_size, seq_len, num_agents, features))

        agent_obs, hideout_obs, timestep_obs, num_agents = x
        # start_location = start_location.to(self.device).float()
        agent_obs = agent_obs.to(self.device).float()

        # agent_obs = torch.cat((agent_obs, location_obs), dim=2)

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        # num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        graph_list = []
        agent_features = []

        for n, batch in zip(num_agents, list(range(batch_size))):
            # assert n.item().is_integer()
            n = int(n.item())
            if n not in self.initialized_graphs:
                g = self.initialize_graph(n)
                self.initialized_graphs[n] = g
            else:
                g = self.initialized_graphs[n]
            h = agent_obs[batch, :, :n, :]
            agent_features.append(h)
            graph_list.append(g)

        pyg_batch = Batch.from_data_list(graph_list).to(self.device)
        agent_features = torch.cat(agent_features, dim=1).contiguous() # seq_len x total_num_agents x features

        for t in range(seq_len):
            pyg_batch.x = agent_features[t, :, :]
            if t == 0:
                h, c = self.gclstm(pyg_batch.x, pyg_batch.edge_index)
            else:
                h, c = self.gclstm(pyg_batch.x, pyg_batch.edge_index, H=h, C=c)
        
        res = global_mean_pool(h, pyg_batch.batch) # [B x hidden]
        res = torch.cat((res, hideout_obs, timestep_obs), dim=-1)
        return res