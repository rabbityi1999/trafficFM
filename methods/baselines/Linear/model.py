"""Direct multi-horizon linear forecasting in the time domain."""
import torch
from torch import nn


class Linear(nn.Module):
    """Map [B,L,N] or [B,L,N,1] to the same layout with horizon H.

    With individual=False, all sensors share a temporal projection.
    With individual=True, each sensor has its own weights and bias.
    """

    def __init__(self, input_len=144, output_len=144, num_nodes=None, individual=False):
        super().__init__()
        if input_len < 1 or output_len < 1:
            raise ValueError('input_len and output_len must be positive')
        if individual and (num_nodes is None or num_nodes < 1):
            raise ValueError('individual projections require num_nodes')
        self.input_len, self.output_len = input_len, output_len
        self.num_nodes, self.individual = num_nodes, individual
        if individual:
            self.weight = nn.Parameter(torch.empty(num_nodes, output_len, input_len))
            self.bias = nn.Parameter(torch.empty(num_nodes, output_len))
            nn.init.uniform_(self.weight, -input_len**-.5, input_len**-.5)
            nn.init.uniform_(self.bias, -input_len**-.5, input_len**-.5)
        else:
            self.projection = nn.Linear(input_len, output_len)

    def forward(self, history):
        four_dimensional = history.ndim == 4
        if four_dimensional:
            if history.shape[-1] != 1:
                raise ValueError('Linear consumes the speed feature only')
            history = history[..., 0]
        if history.ndim != 3 or history.shape[1] != self.input_len:
            raise ValueError('expected [batch,input_len,nodes] or [...,1]')
        if self.num_nodes is not None and history.shape[2] != self.num_nodes:
            raise ValueError('sensor count differs from the model configuration')
        if self.individual:
            result = torch.einsum('bln,nhl->bhn', history, self.weight)
            result = result + self.bias.T.unsqueeze(0)
        else:
            result = self.projection(history.transpose(1, 2)).transpose(1, 2)
        return result.unsqueeze(-1) if four_dimensional else result
