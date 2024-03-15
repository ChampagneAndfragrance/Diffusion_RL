from re import X
from matplotlib.pyplot import xlabel, xlim
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from models.encoders import EncoderRNN
from itertools import combinations, combinations_with_replacement
import torch
from dgl.nn import AvgPooling
import numpy as np

def fully_connected_include_self(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations_with_replacement(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)

def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)

def central_graph(num_nodes):
    # create graph where each node is connected to the central node which represents the agent

    if num_nodes == 1:
        return [], []
    else:
        start_nodes = [0] * (num_nodes - 1)
        end_nodes = list(range(1, num_nodes))
    return torch.tensor(start_nodes), torch.tensor(end_nodes)

# Create graph encoder consisting of an lstm into a graph NN
class GNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden, num_layers=1, last_k_fugitive_detection_bool=False, last_two_detection_with_vel_bool=False, start_location=False):
        super(GNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.last_k_fugitive_detection_bool = last_k_fugitive_detection_bool
        self.last_two_detection_with_vel_bool = last_two_detection_with_vel_bool
        self.start_location = start_location
        self.lstm = EncoderRNN(input_dim, hidden_dim, num_layers)
        # self.location_lstm = EncoderRNN(2, 2, 1)

        self.act = nn.Tanh()
        # nn initialization already occurs in SAGEConv
        self.conv1 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=hidden_dim, aggregator_type='pool', activation=self.act)
        self.conv2 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=gnn_hidden, aggregator_type='pool')

        if self.last_k_fugitive_detection_bool:
            self.linear = nn.Linear(24, gnn_hidden)
        elif self.last_two_detection_with_vel_bool:
            self.linear = nn.Linear(11, gnn_hidden)

        self.avgpool = AvgPooling()
        self.batched_graphs = None
        self.initialized_graphs = dict()

    def initialize_graph(self, num_agents):
        # initialize graph just from 
        s, e = fully_connected(num_agents)
        s = s.to(self.device)
        e = e.to(self.device)
        g = dgl.graph((s, e), num_nodes = num_agents)
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
        if (self.last_k_fugitive_detection_bool or self.last_two_detection_with_vel_bool) and self.start_location:
            agent_obs, hideout_obs, timestep_obs, num_agents, last_k, start_location = x
            last_k = last_k.to(self.device).float()
            start_location = start_location.to(self.device).float()
        elif (self.last_k_fugitive_detection_bool or self.last_two_detection_with_vel_bool) and not self.start_location:
            agent_obs, hideout_obs, timestep_obs, num_agents, last_k = x
            last_k = last_k.to(self.device).float()
        elif (not self.last_k_fugitive_detection_bool) and (not self.last_two_detection_with_vel_bool) and self.start_location:
            agent_obs, hideout_obs, timestep_obs, num_agents, start_location = x
            start_location = start_location.to(self.device).float()
        else:
            agent_obs, hideout_obs, timestep_obs, num_agents = x

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()
        # last_k_fugitive_detections = last_k_fugitive_detections.to(self.device).float()

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
            if n not in self.initialized_graphs:
                g = self.initialize_graph(n)
                self.initialized_graphs[n] = g
            else:
                g = self.initialized_graphs[n]
            h = agent_obs[batch, :, :n, :]
            graph_list.append(g)
            lstm_input.append(h)
        batched_graphs = dgl.batch(graph_list).to(self.device)
        lstm_input = torch.cat(lstm_input, dim=1).contiguous()
        lstm_input = lstm_input.permute(1, 0, 2)
        hn = self.lstm(lstm_input)
        res = self.conv1(batched_graphs, hn)
        res = self.conv2(batched_graphs, res)
        res = self.avgpool(batched_graphs, res)
        # [B x hidden_dim]

        if self.last_k_fugitive_detection_bool:
            # last_k = last_k.view(last_k.size(0), -1)
            last_k = last_k.reshape(last_k.size(0), 24)
            # res = torch.cat((res, last_k), dim=-1)
            res = self.linear(last_k) + res
        elif self.last_two_detection_with_vel_bool:
            last_k = last_k.reshape(last_k.size(0), 11) # original: 11
            res = self.linear(last_k) + res

        # location_lstm_output = self.location_lstm(location_obs)
        res = torch.cat((res, hideout_obs, timestep_obs), dim=-1)

        if self.start_location:
            res = torch.cat((res, start_location), dim=-1)

        # all unknown hideouts
        # unknown_hideouts = np.array([(234, 2082), (1191, 950), (563, 750), (2314, 86), (1119, 1623), (1636, 2136), (602, 1781), (2276, 1007), (980, 118), (2258, 1598)])/2428
        # unknown_hideouts = torch.from_numpy(unknown_hideouts).to(self.device).float().flatten().unsqueeze(0).repeat(batch_size, 1)
        # res = torch.cat((res, unknown_hideouts), dim=-1)

        return res


def compute_similarity(vec_a, vec_b):
    loss = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    res = loss(vec_a, vec_b)
    return res


def NT_XENTLoss(all_vecs, temperature=1.0, time_diff=1.0):
    loss = torch.nn.Softmax(dim=1)
    # NT-Xent loss:
    vals = -torch.log(loss(all_vecs/temperature))

    return vals[:, -1] * (1.0 / time_diff)  # The softmax with respect to the positive embedding (appended at the end)

# Create graph encoder consisting of an lstm into a graph NN
class ContrastiveGNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden, num_layers=1, use_last_k_detections=False):
        super(ContrastiveGNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = EncoderRNN(input_dim, hidden_dim, num_layers)
        # self.lstm_neg = EncoderRNN(input_dim, hidden_dim // 2, num_layers)
        # self.location_lstm = EncoderRNN(2, 2, 1)

        self.act = nn.Tanh()
        # nn initialization already occurs in SAGEConv
        self.conv1 = dglnn.GATv2Conv(
            in_feats=hidden_dim, out_feats=hidden_dim, num_heads=8)
        self.conv2 = dglnn.GATv2Conv(
            in_feats=hidden_dim, out_feats=gnn_hidden, num_heads=1)

        # self.linear = nn.Linear(hidden_dim + 3, 16)
        self.avgpool = AvgPooling()
        self.batched_graphs = None
        # self.initialize_graphs()

        self.initialized_graphs = {n: self.initialize_graph(n) for n in range(83, 136)}
        self.use_last_k_detections = use_last_k_detections

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

        # Compute separate graph convolutions + LSTM for positive and negative sequence of data
        agent_anchor_obs, agent_positive_obs, agent_negative_obs, hideout_obs, timestep_obs, num_agents, \
        last_positive_timestep, last_negative_timestep = x

        agent_anchor_obs = agent_anchor_obs.to(self.device).float()
        agent_positive_obs = agent_positive_obs.to(self.device).float()
        agent_negative_obs = agent_negative_obs.to(self.device).float()

        hideout_obs = hideout_obs.to(self.device).float()
        timestep_obs = timestep_obs.to(self.device).float()

        last_positive_timestep = last_positive_timestep.to(self.device)
        last_negative_timestep = last_negative_timestep.to(self.device)

        batch_size = agent_positive_obs.shape[0]
        seq_len = agent_positive_obs.shape[1]
        # num_agents = agent_positive_obs.shape[2]
        features = agent_positive_obs.shape[3]

        # permuted = agent_obs.permute(0, 2, 1, 3) # (batch_size, num_agents, seq_len, features)
        # hn is of shape (batch_size * num_agents, hidden_dim)
        # .view(batch_size * lstm_input.shape[1], seq_len, features)

        graph_list = []
        anchor_lstm_input = []
        pos_lstm_input = []
        neg_lstm_input = []

        num_negative_samples = agent_negative_obs.shape[1]

        # Total batch size for anchor, positive and negative samples = (batch_size + batch_size + num_negative_samples * batch_size)
        # total batch size = (2 + N) * batch_size
        for n, batch in zip(num_agents, list(range(batch_size))):
            # assert n.item().is_integer()
            n = int(n.item())

            g = self.initialized_graphs[n]
            h_anchor = agent_anchor_obs[batch, :, :n, :]
            h_pos = agent_positive_obs[batch, :, :n, :]
            h_neg = agent_negative_obs[batch, :, :n, :]
            graph_list.append(g)
            anchor_lstm_input.append(h_anchor)
            pos_lstm_input.append(h_pos)
            neg_lstm_input.append(h_neg)

        # Additionally stack graphs so that we can pass anchor, positive and negative samples all at once
        for k in range(batch_size * (num_negative_samples + 1)):
            graph_list.append(g)

        batched_graphs = dgl.batch(graph_list).to(self.device)

        anchor_lstm_input = torch.cat(anchor_lstm_input, dim=1).contiguous()
        pos_lstm_input = torch.cat(pos_lstm_input, dim=1).contiguous()
        neg_lstm_input = torch.cat(neg_lstm_input, dim=2).contiguous()

        anchor_lstm_input = anchor_lstm_input.permute(1, 0, 2)
        anchor_hn = self.lstm(anchor_lstm_input)

        pos_lstm_input = pos_lstm_input.permute(1, 0, 2)
        pos_hn = self.lstm(pos_lstm_input)

        batched_input_shape, batched_feats_shape = pos_hn.shape

        # neg_lstm_input = neg_lstm_input.permute(1, 0, 2)
        neg_lstm_input = neg_lstm_input.permute(2, 0, 1, 3)
        neg_lstm_input = neg_lstm_input.reshape(-1, seq_len, features)
        neg_hn = self.lstm(neg_lstm_input)

        # Concatenate all lstm inputs from anchor, positive and negative observations and pass through the graph at once
        all_res = torch.cat((anchor_hn, pos_hn), dim=0).contiguous()
        all_res = torch.cat((all_res, neg_hn), dim=0).contiguous()

        all_res = self.conv1(batched_graphs, all_res)
        all_res = torch.mean(all_res, dim=1)  # Average of the attention heads
        all_res = self.conv2(batched_graphs, all_res)
        all_res = all_res.squeeze()
        all_res = self.avgpool(batched_graphs, all_res)  # [B x hidden_dim]

        # Unstack from the graph to get the embeddings of anchor, positive and negative observations
        anchor_res = all_res[:batch_size]
        pos_res = all_res[batch_size: 2*batch_size]
        neg_res = all_res[2*batch_size:]


        # anchor_res = self.conv1(batched_graphs, anchor_hn)
        # anchor_res = torch.mean(anchor_res, dim=1)  # Average of the attention heads
        # anchor_res = self.conv2(batched_graphs, anchor_res)
        # anchor_res = anchor_res.squeeze()
        # anchor_res = self.avgpool(batched_graphs, anchor_res)  # [B x hidden_dim]
        #
        # pos_res = self.conv1(batched_graphs, pos_hn)
        # pos_res = torch.mean(pos_res, dim=1)  # Average of the attention heads
        # pos_res = self.conv2(batched_graphs, pos_res)
        # pos_res = pos_res.squeeze()
        # pos_res = self.avgpool(batched_graphs, pos_res)  # [B x hidden_dim]
        #
        # # Reshape and calculate the value of embeddings separately for contrastive learning
        # neg_hn = neg_hn.reshape(-1, batched_input_shape, batched_feats_shape)
        # num_negative_samples = neg_hn.shape[0]
        # neg_embeddings = []
        #
        # all_vecs = []
        #
        # # TODO: @Manisha Unsure how to batch num contrastive samples since the DGL graph is only set up for 84 agents with seq_len
        # for z in range(num_negative_samples):
        #     neg_res = self.conv1(batched_graphs, neg_hn[z])
        #     neg_res = torch.mean(neg_res, dim=1)  # Average of the attention heads
        #     neg_res = self.conv2(batched_graphs, neg_res)
        #     neg_res = neg_res.squeeze()
        #     neg_res = self.avgpool(batched_graphs, neg_res)  # [B x hidden_dim]
        #     all_vecs.append(compute_similarity(anchor_res, neg_res).unsqueeze(0))

        # Compute contrastive loss here
        sim_pos = compute_similarity(anchor_res, pos_res)
        sim_pos = sim_pos.unsqueeze(1)
        sim_neg = compute_similarity(anchor_res.repeat(num_negative_samples, 1), neg_res)
        sim_neg = sim_neg.reshape(-1, num_negative_samples)

        all_vecs = torch.cat((sim_neg, sim_pos), dim=1).contiguous()

        contrastive_loss = NT_XENTLoss(all_vecs)

        res = torch.cat((anchor_res, hideout_obs, timestep_obs), dim=-1)

        # Only pass the embedding of the anchor for the decoder
        return res, contrastive_loss.mean()
