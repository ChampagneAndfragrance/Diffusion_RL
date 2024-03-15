""" Just use the GCLSTM from pytorch geometric for the agents and concat it with the hideout information """

from cgitb import reset
from torch_geometric_temporal.nn.recurrent.gc_lstm import GCLSTM
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.data import Data
import torch_geometric.transforms as T
# from torch_geometric.nn import global_mean_pool
from torch_geometric.graphgym.models.pooling import global_mean_pool
from torch_geometric.nn import HeteroConv, SAGEConv

from itertools import combinations
from torch_geometric.data import HeteroData

def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)

# Create graph encoder consisting of an lstm into a graph NN
class HybridGNN(nn.Module):
    def __init__(self, input_dim, gnn_hidden, output_dim, last_k_fugitive_detection_bool=False, start_location=False):
        super(HybridGNN, self).__init__()
        self.gnn_hidden = gnn_hidden
        self.output_dim = output_dim
        self.last_k_fugitive_detection_bool = last_k_fugitive_detection_bool
        self.start_location = start_location
        self.initialized_graphs = dict()
        self.act = nn.Tanh()

        nodes = ['agents', 'hideout', 'state_summ']
        edges = [
        ('agents', 'to', 'state_summ'), 
        ('hideout', 'to', 'state_summ')]
        # ('timestep', 'to', 'state_summ')]

        self.metadata = (nodes, edges)

        self.gclstm = GCLSTM(in_channels=input_dim, out_channels=gnn_hidden, K=1)
        self.conv = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                                    out_channels=output_dim,
                                                    bias=True) for edge_type in self.metadata[1]})

    @property
    def device(self):
        return next(self.parameters()).device

    def initialize_graph(self, n_agents):
        from_nodes, to_nodes = fully_connected(n_agents)
        edges = torch.stack((from_nodes, to_nodes)).to(torch.long)
        data = Data(edge_index=edges)
        data.num_nodes = n_agents
        return data

    def build_batched_heterogeneous_graph(self, batch_size):
        data=HeteroData()
        # from agent to agent summary node
        agent_indices = torch.arange(0, batch_size)
        hideout_indices = torch.arange(0, batch_size)
        timestep_indices = torch.arange(0, batch_size)
        state_summary_indices = torch.arange(0, batch_size)

        data['agents'].num_nodes = batch_size
        data['hideout'].num_nodes = batch_size
        # data['timestep'].num_nodes = batch_size

        data['state_summ'].num_nodes = batch_size

        data['agents', 'to', 'state_summ'].edge_index = torch.stack((agent_indices, state_summary_indices))
        data['hideout', 'to', 'state_summ'].edge_index = torch.stack((hideout_indices, state_summary_indices))
        # data['timestep', 'to', 'state_summ'].edge_index = torch.stack((timestep_indices, state_summary_indices))

        # if start_location_bool:
        #     data['start_location'].num_nodes = 1
        #     data['start_location', 'to', 'state_summ'].edge_index = torch.zeros((2, 1), dtype=torch.long)

        return data

    def forward(self, x):
        # x is (batch_size, seq_len, num_agents, features)
        # input_tensor = torch.randn((batch_size, seq_len, num_agents, features))

        if self.last_k_fugitive_detection_bool and self.start_location:
            agent_obs, hideout_obs, timestep_obs, num_agents, last_k, start_location = x
            last_k = last_k.to(self.device).float()
            start_location = start_location.to(self.device).float()
        elif self.last_k_fugitive_detection_bool and not self.start_location:
            agent_obs, hideout_obs, timestep_obs, num_agents, last_k = x
            last_k = last_k.to(self.device).float()
        elif not self.last_k_fugitive_detection_bool and self.start_location:
            agent_obs, hideout_obs, timestep_obs, num_agents, start_location = x
            start_location = start_location.to(self.device).float()
        else:
            agent_obs, hideout_obs, timestep_obs, num_agents = x

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
        # res = torch.cat((res, hideout_obs, timestep_obs), dim=-1)
        # instead of using just concatenation, let's use the heterogeneous gnn

        pyg = self.build_batched_heterogeneous_graph(batch_size).to(self.device)
        pyg = T.ToUndirected()(pyg)

        pyg['agents'].x = res
        pyg['hideout'].x = hideout_obs # hideouts don't change over time
        pyg['state_summ'].x = torch.zeros((batch_size, 1), device=self.device)
        # if self.start_location:
        #     pyg['start_location'].x = start_location  # start location also doesn't change over time

        # res = self.conv1(batched_graphs, hn)
        res = self.conv(pyg.x_dict, pyg.edge_index_dict)
        res = {node_type: self.act(feat) for node_type, feat in res.items() }
        # res = torch.cat((res['state_summ'], hideout_obs, timestep_obs), dim=-1)

        res = res['state_summ']

        if self.last_k_fugitive_detection_bool:
            last_k = last_k.view(last_k.size(0), -1)
            res = torch.cat((res, last_k), dim=-1)

        if self.start_location:
            res = torch.cat((res, start_location), dim=-1)
        
        return res