import torch
from torch_geometric.data import HeteroData
from models.gnn.hetero_gc_lstm import HeteroGCLSTM # Until the pytorch geometric temporal model is fixed - just created a copy here
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.nn import HeteroConv, SAGEConv

def construct_het_graph(num_agents, num_hideouts, start_location_bool=False):
    data=HeteroData()
    # from agent to agent summary node
    agent_indices = torch.arange(0, num_agents)
    agent_summary_index = torch.tensor([0] * num_agents) # torch.zeros didn't work - error with dgl

    hideout_indices = torch.arange(0, num_hideouts)
    hideout_summary_index = torch.tensor([0] * num_hideouts)

    data['agent'].num_nodes = num_agents
    data['hideout'].num_nodes = num_hideouts
    data['hideout_summ'].num_nodes = 1
    data['state_summ'].num_nodes = 1
    data['timestep'].num_nodes = 1
    data['agent_summ'].num_nodes = 1

    data['agent', 'to', 'agent_summ'].edge_index = torch.stack((agent_indices, agent_summary_index))
    data['hideout', 'to', 'hideout_summ'].edge_index = torch.stack((hideout_indices, hideout_summary_index))
    data['hideout_summ', 'to', 'state_summ'].edge_index = torch.zeros((2, 1), dtype=torch.long)
    data['agent_summ', 'to', 'state_summ'].edge_index = torch.zeros((2, 1), dtype=torch.long)
    data['timestep', 'to', 'state_summ'].edge_index = torch.zeros((2, 1), dtype=torch.long)

    if start_location_bool:
        data['start_location'].num_nodes = 1
        data['start_location', 'to', 'state_summ'].edge_index = torch.zeros((2, 1), dtype=torch.long)

    return data

class HeteroLSTM(nn.Module):
    def __init__(self, gnn_hidden, num_layers=1, start_location_bool=False):
        super(HeteroLSTM, self).__init__()
        self.start_location_bool = start_location_bool

        self.initialized_graphs = dict()
        metadata = (['agent', 'hideout', 'timestep', 'agent_summ', 'hideout_summ', 'state_summ'], 
        [('agent', 'to', 'agent_summ'), 
        ('hideout', 'to', 'hideout_summ'), 
        ('hideout_summ', 'to', 'state_summ'), 
        ('agent_summ', 'to', 'state_summ'), 
        ('timestep', 'to', 'state_summ'), 
        ('agent_summ', 'rev_to', 'agent'), 
        ('hideout_summ', 'rev_to', 'hideout'), 
        ('state_summ', 'rev_to', 'hideout_summ'), 
        ('state_summ', 'rev_to', 'agent_summ'), 
        ('state_summ', 'rev_to', 'timestep')])

        if start_location_bool:
            metadata.append(('start_location', 'to', 'state_summ'))
            metadata.append(('state_summ', 'rev_to', 'start_location'))
            in_channels_dict = {'agent': 8, 'hideout': 2, 'timestep': 1, 'agent_summ': 1, 'hideout_summ': 1, 'state_summ': 1, 'start_location': 1}
        else:
            in_channels_dict = {'agent': 8, 'hideout': 2, 'timestep': 1, 'agent_summ': 1, 'hideout_summ': 1, 'state_summ': 1}

        
        self.hetero_lstm = HeteroGCLSTM(in_channels_dict = in_channels_dict, out_channels = gnn_hidden, metadata = metadata).to("cuda")
        self.conv = HeteroConv({edge_type: SAGEConv(in_channels=(-1, -1),
                                            out_channels=gnn_hidden,
                                            bias=True) for edge_type in metadata[1]}).to("cuda")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        # x is (batch_size, seq_len, num_agents, features)
        # input_tensor = torch.randn((batch_size, seq_len, num_agents, features))

        # agent_obs, hideout_obs, timestep_obs = x
        
        if self.start_location_bool:
            agent_obs, hideout_obs, timestep_obs, num_agents, start_location = x
            start_location = start_location.to(self.device).float()
        else:
            agent_obs, hideout_obs, timestep_obs, num_agents = x
        agent_obs = agent_obs.to(self.device).float()

        # agent_obs = torch.cat((agent_obs, location_obs), dim=2)

        hideout_obs = hideout_obs.to(self.device).float() # batch x 2
        timestep_obs = timestep_obs.to(self.device).float() # batch x seq_len

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        features = agent_obs.shape[3]

        graph_list = []
        agent_features = []

        for n, batch in zip(num_agents, list(range(batch_size))):
            n = int(n.item())

            if n not in self.initialized_graphs:
                pyg = construct_het_graph(n, 1, self.start_location_bool)
                self.initialized_graphs[n] = pyg
            else:
                pyg = self.initialized_graphs[n]

            h = agent_obs[batch, :, :n, :]
            agent_features.append(h)
            graph_list.append(pyg)

        pyg_batch = Batch.from_data_list(graph_list).to(self.device)
        pyg_batch = T.ToUndirected()(pyg_batch)

        agent_features = torch.cat(agent_features, dim=1).contiguous() # seq_len x total_num_agents x features

        for t in range(seq_len):
            pyg_batch['agent'].x = agent_features[t, :, :]
            pyg_batch['hideout'].x = hideout_obs # hideouts don't change over time
            pyg_batch['timestep'].x = timestep_obs[:, t].unsqueeze(1)
            pyg_batch['agent_summ'].x = torch.zeros((batch_size, 1), device=self.device)
            pyg_batch['hideout_summ'].x = torch.zeros((batch_size, 1), device=self.device)
            pyg_batch['state_summ'].x = torch.zeros((batch_size, 1), device=self.device)
            if self.start_location_bool:
                pyg_batch['start_location'].x = start_location 
            if t == 0:
                h_dict, c_dict = self.hetero_lstm(pyg_batch.x_dict, pyg_batch.edge_index_dict)
            else:
                h_dict, c_dict = self.hetero_lstm(pyg_batch.x_dict, pyg_batch.edge_index_dict, h_dict=h_dict, c_dict=c_dict)

        # out_dict = self.conv(h_dict, pyg_batch.edge_index_dict)
        return h_dict['state_summ']