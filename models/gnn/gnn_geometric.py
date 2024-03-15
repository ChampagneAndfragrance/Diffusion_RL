from re import X
from matplotlib.pyplot import xlabel, xlim
import torch
import torch.nn as nn
from models.encoders import EncoderRNN
from itertools import combinations
import torch

def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)

# Create graph encoder consisting of an lstm into a graph NN
class GNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden, num_layers=1):
        super(GNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = EncoderRNN(input_dim, hidden_dim, num_layers)
        # self.location_lstm = EncoderRNN(2, 2, 1)

        self.act = nn.Tanh()
        # nn initialization already occurs in SAGEConv
        self.conv1 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=hidden_dim, aggregator_type='pool', activation = self.act)
        self.conv2 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=gnn_hidden, aggregator_type='pool')

        # self.linear = nn.Linear(hidden_dim + 3, 16)
        self.avgpool = AvgPooling()
        self.batched_graphs = None
        # self.initialize_graphs()

        self.initialized_graphs = {n: self.initialize_graph(n) for n in range(83, 95)}

    def initialize_graph(self, num_agents):
        # initialize graph just from 
        s, e = fully_connected(num_agents)
        s = s.to(self.device)
        e = e.to(self.device)
        g = dgl.graph((s, e))
        g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
        return g


    def initialize_graphs(self, batch_size, num_agents):
        # quick hack to just speed things up, assume same graph for all batches
        # num_agents = 81
        # batch_size = 128
        graph_list = []
        s, e = fully_connected(num_agents)
        s = s.to(self.device)
        e = e.to(self.device)

        for _ in range(batch_size):
            g = dgl.graph((s, e))
            g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
            graph_list.append(g)
        self.batched_graphs = dgl.batch(graph_list)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        # x is (batch_size, seq_len, num_agents, features)
        # input_tensor = torch.randn((batch_size, seq_len, num_agents, features))

        # agent_obs, hideout_obs, timestep_obs = x
        
        agent_obs, hideout_obs, timestep_obs, num_agents = x
        agent_obs = agent_obs.to(self.device).float()

        # agent_obs = torch.cat((agent_obs, location_obs), dim=2)

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        # num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        # permuted = agent_obs.permute(0, 2, 1, 3) # (batch_size, num_agents, seq_len, features)
        # hn is of shape (batch_size * num_agents, hidden_dim)
        # .view(batch_size * lstm_input.shape[1], seq_len, features)

        graph_list = []
        lstm_input = []

        for n, batch in zip(num_agents, list(range(batch_size))):
            # assert n.item().is_integer()
            n = int(n.item())

            g = self.initialized_graphs[n]
            h = agent_obs[batch, :, :n, :]
            graph_list.append(g)
            lstm_input.append(h)
        batched_graphs = dgl.batch(graph_list).to(self.device)
        lstm_input = torch.cat(lstm_input, dim=1).contiguous()
        lstm_input = lstm_input.permute(1, 0, 2)
        hn = self.lstm(lstm_input)
        

        # if batch_size != 128:
        #     graph_list = []
        #     s, e = fully_connected(num_agents)
        #     s = s.to(self.device)
        #     e = e.to(self.device)

        #     for _ in range(batch_size):
        #         g = dgl.graph((s, e))
        #         g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
        #         graph_list.append(g)
        #     batched_graphs = dgl.batch(graph_list)
        # else:
        #     # eventually we will create different graphs for different agent setups and connection types
        #     # batched_graphs = dgl.batch(graph_list)
        #     if self.batched_graphs is None:
        #         self.initialize_graphs(batch_size, num_agents)
        #     batched_graphs = self.batched_graphs.to(self.device)

        
        res = self.conv1(batched_graphs, hn)
        res = self.conv2(batched_graphs, res)
        res = self.avgpool(batched_graphs, res)
        # [B x hidden_dim]

        # location_lstm_output = self.location_lstm(location_obs)
        res = torch.cat((res, hideout_obs, timestep_obs), dim=-1)
        # return self.linear(res)

        return res