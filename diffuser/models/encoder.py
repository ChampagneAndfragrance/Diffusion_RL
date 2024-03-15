import torch.nn as nn
import torch.nn.functional as F
import torch

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):

        super(EncoderRNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, hidden_state=None):
        if type(x) == torch.nn.utils.rnn.PackedSequence:
            batch_size = x.sorted_indices.size(0)
        else:
            batch_size = x.size(0)
            
        x = x.to(self.device).float()
        if hidden_state is None:
            # Initializing the hidden state for the first input with zeros
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)

            # Initializing the cell state for the first input with zeros
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_().to(self.device)
        else:
            h0, c0 = hidden_state
        out, (hn, cn) = self.lstm(x, (h0, c0))

        hn = hn.view(-1, hn.shape[-1])
        return hn
    
    def flatten_parameters(self):
        self.lstm.flatten_parameters()


