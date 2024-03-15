"""
Create a Graph+LSTM encoder with CPC structure
- Each data sample (blue observation) is matched with future (blue or) red states
- The two samples may not be of the same dimensions (use linear transformation)
- All other samples in the minibatch are considered as negative samples
- CPC uses InfoNCE loss
- Here I will assume that both the current blue and future states are available in the forward pass
"""

from re import X
from matplotlib.pyplot import xlabel, xlim
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from models.encoders import EncoderRNN
from itertools import combinations
import torch
from dgl.nn import AvgPooling


LARGE_NUM = 1e-9

def fully_connected(num_nodes):
    # create fully connected graph
    test_list = range(num_nodes)
    edges = list(combinations(test_list, 2))
    start_nodes = [i[0] for i in edges]
    end_nodes = [i[1] for i in edges]
    return torch.tensor(start_nodes), torch.tensor(end_nodes)


class CPCGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden,
                 num_timesteps=12, contrastive_latent_dim=16, num_layers=1, use_last_k_detections=False,
                 autoregressive=False):
        super(CPCGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.contrastive_latent_dim = contrastive_latent_dim
        self.lstm = EncoderRNN(input_dim, hidden_dim, num_layers)
        # self.location_lstm = EncoderRNN(2, 2, 1)

        self.act = nn.Tanh()
        # nn initialization already occurs in SAGEConv
        self.conv1 = dglnn.GATv2Conv(
            in_feats=hidden_dim, out_feats=hidden_dim, num_heads=8)
        self.conv2 = dglnn.GATv2Conv(
            in_feats=hidden_dim, out_feats=gnn_hidden, num_heads=1)

        self.num_timesteps = num_timesteps  # Number of timesteps in to the future that we are predicting
        # (= num prediction heads in the decoder)
        self.Wk = nn.ModuleList([nn.Linear(gnn_hidden, contrastive_latent_dim) for _ in range(num_timesteps)])

        # Project future states into latent space
        self.linear = nn.Linear(2, contrastive_latent_dim)
        self.avgpool = AvgPooling()
        self.batched_graphs = None

        self.initialized_graphs = dict()
        self.use_last_k_detections = use_last_k_detections
        self.autoregressive = autoregressive

    def initialize_graph(self, num_agents):
        # initialize graph just from
        s, e = fully_connected(num_agents)
        s = s.to(self.device)
        e = e.to(self.device)
        g = dgl.graph((s, e))
        g = dgl.add_reverse_edges(dgl.to_homogeneous(g))
        return g

    @property
    def device(self):
        return next(self.parameters()).device

    def InfoNCELoss(self, true_future_states, preds, normalize=True):
        batch_size = true_future_states.shape[1]
        nce_loss = 0
        if normalize:
            true_future_states = F.normalize(true_future_states, dim=-1)
            preds = F.normalize(preds, dim=-1)

        for i in range(self.num_timesteps):
            total = torch.mm(true_future_states[i], torch.transpose(preds[i], 0, 1))  # Batch x Batch
            log_softmax = nn.LogSoftmax(dim=-1)
            nce_loss += torch.sum(torch.diag(log_softmax(total)))

        return - nce_loss/(batch_size * self.num_timesteps)

    def forward(self, x):
        if self.autoregressive:
            agent_obs, future_obs, hideout_obs, timestep_obs, num_agents, decoder_input, targets = x
            decoder_input = decoder_input.to(self.device).float()
            targets = targets.to(self.device).float()
        else:
            agent_obs, future_obs, hideout_obs, timestep_obs, num_agents = x

        # agent_obs: [batch_size, seq_len, num_agents, num_feats]
        agent_obs = agent_obs.to(self.device).float()

        # future_obs: [batch_size, num_timesteps, num_feats_future]
        future_obs = future_obs.to(self.device).float()

        # Other env info ...
        # hideout_obs: [batch_size, 2]
        hideout_obs = hideout_obs.to(self.device).float()

        # timestep_obs: [batch_size, 1]
        timestep_obs = timestep_obs.to(self.device).float()

        # Extract dimensions of the current dataset
        batch_size = agent_obs.shape[0]
        seq_len = agent_obs.shape[1]
        # num_agents = agent_obs.shape[2]
        features = agent_obs.shape[3]

        num_future_features = future_obs.shape[-1]

        # Create batched graphs for message passing in DGL
        graph_list = []
        agent_lstm_input = []  # to concat all agent obs inputs

        for n, batch in zip(num_agents, list(range(batch_size))):
            # assert n.item().is_integer()
            n = int(n.item())

            if n not in self.initialized_graphs:
                g = self.initialize_graph(n)
                self.initialized_graphs[n] = g
            else:
                g = self.initialized_graphs[n]

            h_agent = agent_obs[batch, :, :n, :]
            graph_list.append(g)
            agent_lstm_input.append(h_agent)

        batched_graphs = dgl.batch(graph_list).to(self.device)
        agent_lstm_input = torch.cat(agent_lstm_input, dim=1).contiguous()

        # Get hidden vectors after LSTM for anchor and augmented obs
        agent_lstm_input = agent_lstm_input.permute(1, 0, 2)
        agent_hn = self.lstm(agent_lstm_input)

        # Use the hidden state from LSTM as node embedding in GraphConv
        agent_res = self.conv1(batched_graphs, agent_hn)
        agent_res = torch.mean(agent_res, dim=1)
        agent_res = self.conv2(batched_graphs, agent_res)
        agent_res = agent_res.squeeze(dim=1)
        # agent_context: [batch_size, gnn_hidden]
        agent_context = self.avgpool(batched_graphs, agent_res)  # equivalent to c_t in CPC

        projected_future_obs = torch.empty((self.num_timesteps, batch_size, self.contrastive_latent_dim)).to(self.device).float()
        pred = torch.empty((self.num_timesteps, batch_size, self.contrastive_latent_dim)).to(self.device).float()
        # Predict the future states from the context vector
        for i in torch.arange(0, self.num_timesteps):
            linear = self.Wk[i]
            pred[i] = linear(agent_context)
            # Project future states into latent space
            projected_future_obs[i] = self.linear(future_obs[:, i, :])

        cpc_loss = self.InfoNCELoss(true_future_states=projected_future_obs, preds=pred)

        agent_context = torch.cat((agent_context, hideout_obs, timestep_obs), dim=-1)

        if self.autoregressive:
            return agent_context, cpc_loss, decoder_input, targets

        return agent_context, cpc_loss
