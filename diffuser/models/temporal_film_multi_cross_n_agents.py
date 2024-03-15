from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from torchsummary import summary

from diffuser.models.temporal import TemporalUnet
from diffuser.models.encoder import EncoderRNN

# from diffusion_policy.model.diffusion.conv1d_components import (
    # Downsample1d, Upsample1d, Conv1dBlock)
# from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

from diffuser.models.helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        cond = cond.float()
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class Head(nn.Module):
    def __init__(self, n_embd, head_size, bias):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=bias)
        self.query = nn.Linear(n_embd, head_size, bias=bias)
        self.value = nn.Linear(n_embd, head_size, bias=bias)

        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, x):
        # n_agents x batch_size x horizon x num_features
        n, b, h, f = x.shape

        keys = self.key(x) 
        queries = self.query(x) 
        values = self.value(x)

        k = keys.transpose(-2, -1) # [n x b x f x h]
        alphas = torch.matmul(queries.unsqueeze(0), k.unsqueeze(1)) * (f ** -0.5) # [n x n x b x h x h]
        alphas = alphas.softmax(dim=-1) # [n x n x b x h x h]
        out = torch.matmul(alphas, values.unsqueeze(1)) # [n x n x b x h x h]

        # out is [k_i * v_i, q_i ...]
        o = out.sum(dim=0) # [n x b x h x h]
        return o

class CrossAttentionBlock(nn.Module):
    def __init__(self, num_features, num_heads, dropout):
        super().__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.head_size = num_features // num_heads
        self.heads = nn.ModuleList([Head(num_features, self.head_size, bias=False) for _ in range(num_heads)])
        self.proj = nn.Linear(num_features, num_features)
        self.dropout = nn.Dropout(dropout)

        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, n_agents):
        # x shape : [batch_size*n_agents x num_features x horizon]
        b, f, h = x.shape
        b = b // n_agents
        x = x.reshape(n_agents, b, f, h) # [n_agents x batch_size x num_features x horizon]
        x = einops.rearrange(x, 'n b f h -> n b h f')
        y = torch.cat([h(x) for h in self.heads], dim=-1) # could do another optimization by doing all heads at once
        y = self.dropout(self.proj(y))

        y = einops.rearrange(y, 'n b h f -> n b f h')
        y = y.reshape(n_agents * b, f, h)
        # z = torch.cat([y[0], y[1]], dim=0)
        # assert torch.allclose(y1, z)
        return y


class ConditionalUnet1DSeparateCross(nn.Module):
    """ This model assumes the two tracks come in as separate LSTM tracks"""
    def __init__(self, 
        # input_dim,
        dim = 32,
        transition_dim = 32, 
        horizon = None,
        cond_dim = None,
        local_cond_dim=None,
        global_cond_dim=None,
        lstm_out_dim = 32,
        # diffusion_step_embed_dim=256,
        # down_dims=[256,512,1024],
        # down_dims = [32, 64, 128, 256],
        dim_mults=(1, 2, 4, 8),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        lstm_dim = 3, 
        num_agents = 3,
        bidirectional = False
        ):
        super().__init__()
        self.num_agents = num_agents
        diffusion_step_embed_dim = dim
        input_dim = transition_dim

        self.horizon = horizon

        down_dims = [dim * mult for mult in dim_mults]
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        if lstm_out_dim is not None:
            cond_dim += lstm_out_dim
        #     global_feature_cond_dim = lstm_out_dim + cond_dim
        # else:
        #     global_feature_cond_dim = cond_dim
        
        self.lstm = EncoderRNN(input_dim=lstm_dim, hidden_dim = lstm_out_dim, num_layers = 1, bidirectional=bidirectional)
        self.lstm.flatten_parameters()

        self.red_linear = nn.Sequential(
            nn.Linear(3, lstm_out_dim),
            nn.Mish(),
            nn.Linear(lstm_out_dim, lstm_out_dim),
            nn.Mish()
        )

        self.linear = nn.Linear(lstm_out_dim, lstm_out_dim)
        self.mish = nn.Mish()

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]

        self.mid_one = ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
        
        self.mid_two = ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        self.attention_one = CrossAttentionBlock(dim_out, 4, dropout=0.0)
        self.attention_two = CrossAttentionBlock(mid_dim, 4, dropout=0.0)

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )



    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        assert sample.shape[-1] % 2 == 0, "input_dim must be even"
        num_agents = sample.shape[-1] // 2

        sample = einops.rearrange(sample, 'b h t -> b t h')
        sample = torch.cat([sample[:, i:i+2, :] for i in range(0, num_agents*2, 2)], axis=0) # (B*n_agents, T, input_dim) split the tracks into separate batches

        # 1. time
        # timesteps = torch.cat([timestep, timestep], axis=0)
        timesteps = torch.cat([timestep for _ in range(num_agents)], axis=0)
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            if 'hideouts' in global_cond.keys():
                hideouts_feature = torch.cat([global_cond['hideouts'] for _ in range(num_agents)], axis=0)
                global_feature = torch.cat([hideouts_feature, global_feature], axis=-1)

            detections_encoded = []
            for i in range(num_agents):
                one_hot = torch.zeros(num_agents)
                one_hot[i] = 1
                if i == 0: # red agent
                    # encode the red agent information into a separate network
                    red_agent_info = self.red_linear(global_cond['red_start'].float())

                    # add one hot info to red_agent_info
                    one_hot = einops.repeat(one_hot, 'n -> b n', b=red_agent_info.shape[0]).to(red_agent_info.device)
                    red_agent_info = torch.cat([red_agent_info, one_hot], axis=-1)
                    detections_encoded.append(red_agent_info)
                else:
                    lstm_output = self.lstm(global_cond[f'd{i-1}'])

                    # add one hot info to lstm_output
                    one_hot = einops.repeat(one_hot, 'n -> b n', b=lstm_output.shape[0]).to(lstm_output.device)
                    lstm_output = torch.cat([lstm_output, one_hot], axis=-1)
                    detections_encoded.append(lstm_output)
            
            detections_encoded = torch.cat(detections_encoded, axis=0)
            global_feature = torch.cat([detections_encoded, global_feature], axis=-1)
        
        if 'local_cond' in global_cond.keys():
            local_cond = global_cond['local_cond']
        else:
            local_cond = None

        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        x = self.attention_one(x, num_agents)
        x = self.mid_one(x, global_feature)
        x = self.attention_two(x, num_agents)
        x = self.mid_two(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        batch_size = x.shape[0] // num_agents
        x = torch.cat([x[i:i+batch_size, :, :] for i in range(0, batch_size*num_agents, batch_size)], axis=-1) # (B, T, input_dim)

        return x

if __name__ == "__main__":
    pass