import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.individual = individual
        self.channels = enc_in

        if self.individual:
            self.Linear = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, *args):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = []
            for i in range(self.channels):
                output.append(self.Linear[i](x[:, :, i]))
            x = torch.stack(output, dim=2)
        else:
            x = x.permute(0, 2, 1)
            x = self.Linear(x)
            x = x.permute(0, 2, 1)

        return x  # [Batch, Output length, Channel]
