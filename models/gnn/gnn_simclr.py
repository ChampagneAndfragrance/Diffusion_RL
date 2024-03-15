"""
Create a Graph+LSTM encoder with SimCLR structure
- Each data sample in a mini-batch is augmented (2N samples)
- For each data point (i), the corresponding augmented sample (j) is the positive sample
- All other samples in the minibatch are considered as negative samples
- The SimCLR loss = L(i, j) + L(j, i)
- Here I will assume that we get the mini-batch of original data and the augmented versions in the forward
- Note that in the original SimCLR paper, they actually use two augmentations of the data sample (i.e., both i and
j are augmented samples). Here I am considering i as the original sample

xi --> LSTM --> GraphConv --> Linear --> zi
xj --> LSTM --> GraphConv --> Linear --> zj

L(i, j) = - log( (exp (sim(zi, zj)/t)) / sum(exp (sim(zi, zk)/t)))  for k!= i
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


class SimCLRGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_hidden,
                 contrastive_loss_dim=32, num_layers=1, use_last_k_detections=False,
                 autoregressive=True):
        super(SimCLRGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = EncoderRNN(input_dim, hidden_dim, num_layers)
        # self.location_lstm = EncoderRNN(2, 2, 1)

        self.act = nn.Tanh()
        # nn initialization already occurs in SAGEConv
        self.conv1 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=hidden_dim, aggregator_type='pool', activation=self.act)
        self.conv2 = dglnn.SAGEConv(
            in_feats=hidden_dim, out_feats=gnn_hidden, aggregator_type='pool')

        self.linear = nn.Linear(gnn_hidden, contrastive_loss_dim)
        self.avgpool = AvgPooling()
        self.batched_graphs = None

        self.initialized_graphs = {n: self.initialize_graph(n) for n in range(83, 95)}
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

    def NT_XEntLoss(self, anchor_embed, aug_embed, temp=0.5, normalize=True):
        """
        anchor_embed: [batch_size, feat_dim]
        aug_embed: [batch_size, feat_dim]
        Create a matrix (batch_size x 2*batch_size) of similarity embeddings:
            | sim(zi_1, zi_1), sim(zi_1, zi_2) ..... sim(zi_1, zi_N),  sim(zi_1, zj_1), .... sim(zi_1, zj_N)|
            | sim(zi_2, zj_1), sim(zi_2, zi_2) ..... sim(zi_2, zi_N),  sim(zi_2, zj_1), .... sim(zi_2, zj_N)|
            |                                                                                               |
            |                                                                                               |
            | sim(zi_1, zj_1), sim(zi_N, zj_2) ..... sim(zi_N, zi_N),  sim(zi_N, zj_1), .... sim(zi_N, zj_N)|

        Do not consider the diagonal values and take softmax row-wise to compute NT-XENT loss

        Code adapted from the official SimCLR repo: https://github.com/google-research/simclr/blob/master/tf2/objective.py
        """
        batch_size = anchor_embed.shape[0]

        if normalize:
            anchor_embed = F.normalize(anchor_embed, dim=1)
            aug_embed = F.normalize(aug_embed, dim=1)

        masks = F.one_hot(torch.arange(batch_size), batch_size).to(self.device).float()
        labels = F.one_hot(torch.arange(batch_size), 2 * batch_size).to(self.device).float()

        # To compute similarity matrix
        # logits_aa = similarity of anchor with anchor
        # all logits have shape [batch_size, batch_size]
        logits_aa = torch.tensordot(anchor_embed, anchor_embed.T, dims=1)/temp
        logits_aa = logits_aa - masks * LARGE_NUM  # to exclude diagonal elements in softmax calc.
        logits_bb = torch.tensordot(aug_embed, aug_embed.T, dims=1)/temp
        logits_bb = logits_bb - masks * LARGE_NUM  # to exclude diagonal elements in softmax calc.
        logits_ab = torch.tensordot(anchor_embed, aug_embed.T, dims=1) / temp
        logits_ba = torch.tensordot(aug_embed, anchor_embed.T, dims=1) / temp

        # Stack the two matrices to form the similarity matrix
        loss_a = F.cross_entropy(input=torch.cat((logits_ab, logits_aa), dim=1),
                                 target=labels)

        loss_b = F.cross_entropy(input=torch.cat((logits_ba, logits_bb), dim=1),
                                 target=labels)

        loss = loss_a + loss_b

        return loss.mean()

    def forward(self, x):
        if self.autoregressive:
            # x is batch_size of agent obs, augmented obs, other env info (hideout obs, timestep obs, num agents)
            agent_anchor_obs, agent_augmented_obs, hideout_obs, timestep_obs, num_agents, input, target = x
            input = input.to(self.device).float()
            target = target.to(self.device).float()

        elif (len(x) == 5):
            # x is batch_size of agent obs, augmented obs, other env info (hideout obs, timestep obs, num agents)
            agent_anchor_obs, agent_augmented_obs, hideout_obs, timestep_obs, num_agents, = x

        else:
            # other can be time_diff (between the last detected timestep and the current timestep)
            agent_anchor_obs, agent_augmented_obs, hideout_obs, timestep_obs, num_agents, other = x

        # agent_anchor_obs: [batch_size, seq_len, num_agents, num_feats]
        agent_anchor_obs = agent_anchor_obs.to(self.device).float()

        # agent_anchor_obs: [batch_size, seq_len, num_agents, num_feats]
        agent_augmented_obs = agent_augmented_obs.to(self.device).float()

        # Other env info ...
        # hideout_obs: [batch_size, 2]
        hideout_obs = hideout_obs.to(self.device).float()

        # timestep_obs: [batch_size, 1]
        timestep_obs = timestep_obs.to(self.device).float()

        # Extract dimensions of the current dataset
        batch_size = agent_anchor_obs.shape[0]
        seq_len = agent_anchor_obs.shape[1]
        # num_agents = agent_anchor_obs.shape[2]
        features = agent_anchor_obs.shape[3]

        # Create batched graphs for message passing in DGL
        graph_list = []
        anchor_lstm_input = []  # to concat all anchor inputs
        aug_lstm_input = []  # to concat all augmented inputs

        for n, batch in zip(num_agents, list(range(batch_size))):
            # assert n.item().is_integer()
            n = int(n.item())
            if n not in self.initialized_graphs:
                g = self.initialize_graph(n)
                self.initialized_graphs[n] = g
            else:
                g = self.initialized_graphs[n]
            h_anchor = agent_anchor_obs[batch, :, :n, :]
            h_aug = agent_augmented_obs[batch, :, :n, :]
            graph_list.append(g)
            anchor_lstm_input.append(h_anchor)
            aug_lstm_input.append(h_aug)

        batched_graphs = dgl.batch(graph_list).to(self.device)

        anchor_lstm_input = torch.cat(anchor_lstm_input, dim=1).contiguous()
        aug_lstm_input = torch.cat(aug_lstm_input, dim=1).contiguous()

        # Get hidden vectors after LSTM for anchor and augmented obs
        anchor_lstm_input = anchor_lstm_input.permute(1, 0, 2)
        anchor_hn = self.lstm(anchor_lstm_input)

        aug_lstm_input = aug_lstm_input.permute(1, 0, 2)
        aug_hn = self.lstm(aug_lstm_input)

        # Use the hidden state from LSTM as node embedding in GraphConv
        # For anchor ...
        anchor_res = self.conv1(batched_graphs, anchor_hn)
        anchor_res = self.conv2(batched_graphs, anchor_res)
        anchor_res = self.avgpool(batched_graphs, anchor_res)

        # For augmented data ...
        aug_res = self.conv1(batched_graphs, aug_hn)
        aug_res = self.conv2(batched_graphs, aug_res)
        aug_res = self.avgpool(batched_graphs, aug_res)

        # Pass in the encoder (LSTM + Graph) embeddings into the linear projection head
        # SimCLR paper claims that computing the contrastive loss after the projection
        # is more beneficial than just using the z
        # Note that this projection head is not used later (i.e., in test time).

        anchor_z = F.relu(self.linear(anchor_res))
        aug_z = F.relu(self.linear(aug_res))

        contrastive_loss = self.NT_XEntLoss(anchor_z, aug_z)

        res = torch.cat((anchor_res, hideout_obs, timestep_obs), dim=-1)

        # Only pass the embedding of the anchor for the decoder
        if self.autoregressive:
            return res, contrastive_loss, input, target
        return res, contrastive_loss


    def predict(self, x):
        """
        Eval function that does not take in augmented observations
        :param x:
        :return:
        """
        if self.autoregressive:
            # x is batch_size of agent obs, augmented obs, other env info (hideout obs, timestep obs, num agents)
            agent_anchor_obs, _, hideout_obs, timestep_obs, num_agents, input, target = x
            input = input.to(self.device).float()
            target = target.to(self.device).float()

        elif (len(x) == 5):
            # x is batch_size of agent obs, augmented obs, other env info (hideout obs, timestep obs, num agents)
            agent_anchor_obs, _, hideout_obs, timestep_obs, num_agents = x

        else:
            # other can be time_diff (between the last detected timestep and the current timestep)
            agent_anchor_obs, _, hideout_obs, timestep_obs, num_agents, other = x

        # agent_anchor_obs: [batch_size, seq_len, num_agents, num_feats]
        agent_anchor_obs = agent_anchor_obs.to(self.device).float()


        # Other env info ...
        # hideout_obs: [batch_size, 2]
        hideout_obs = hideout_obs.to(self.device).float()

        # timestep_obs: [batch_size, 1]
        timestep_obs = timestep_obs.to(self.device).float()

        # Extract dimensions of the current dataset
        batch_size = agent_anchor_obs.shape[0]
        seq_len = agent_anchor_obs.shape[1]
        # num_agents = agent_anchor_obs.shape[2]
        features = agent_anchor_obs.shape[3]

        # Create batched graphs for message passing in DGL
        graph_list = []
        anchor_lstm_input = []  # to concat all anchor inputs

        for n, batch in zip(num_agents, list(range(batch_size))):
            # assert n.item().is_integer()
            n = int(n.item())

            g = self.initialized_graphs[n]
            h_anchor = agent_anchor_obs[batch, :, :n, :]
            graph_list.append(g)
            anchor_lstm_input.append(h_anchor)

        batched_graphs = dgl.batch(graph_list).to(self.device)

        anchor_lstm_input = torch.cat(anchor_lstm_input, dim=1).contiguous()

        # Get hidden vectors after LSTM for anchor and augmented obs
        anchor_lstm_input = anchor_lstm_input.permute(1, 0, 2)
        anchor_hn = self.lstm(anchor_lstm_input)


        # Use the hidden state from LSTM as node embedding in GraphConv
        # For anchor ...
        anchor_res = self.conv1(batched_graphs, anchor_hn)
        anchor_res = self.conv2(batched_graphs, anchor_res)
        anchor_res = self.avgpool(batched_graphs, anchor_res)

        # Pass in the encoder (LSTM + Graph) embeddings into the linear projection head
        # SimCLR paper claims that computing the contrastive loss after the projection
        # is more beneficial than just using the z
        # Note that this projection head is not used later (i.e., in test time).

        contrastive_loss = 0

        res = torch.cat((anchor_res, hideout_obs, timestep_obs), dim=-1)

        # Only pass the embedding of the anchor for the decoder
        if self.autoregressive:
            return res, contrastive_loss, input, target
        return res, contrastive_loss
