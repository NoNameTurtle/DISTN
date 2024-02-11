"""
Disentangled Integrated Spatio-temporal Network (DISTN) as illustrated in the Methodology section
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, reduce

from nns.layers.distn_layers import diff_1st, MLP, CausalGraphGenerator, STConvEnc, DirectedGraphConvEnc


class HFSegEncDec(nn.Module):
    def __init__(self,
                 seq_len,
                 horizon,
                 num_nodes,
                 st_conv,
                 seg_len=8,
                 d_embed=64):
        super().__init__()

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.seg_len = seg_len
        self.freq = math.ceil(self.seq_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.horizon / self.seg_len)  # segment number of output
        self.padding_len = self.seg_len * self.freq - self.seq_len

        self.encoder = nn.Linear(seg_len, d_embed)
        self.st_conv = st_conv
        self.predict = nn.Linear(self.freq, self.step)
        self.decoder = nn.Linear(d_embed, seg_len)

    def forward(self, x, adj):
        B, L, N = x.shape  # L = seq_len

        # slicing
        res = torch.cat((x[:, L - self.padding_len:, :], x), dim=1)  # pad the beginning of the sequence
        res = res.chunk(self.freq, dim=1)  # tuple of (B, seg_len, N)
        res = rearrange(torch.stack(res, dim=1), 'b f p n -> b f n p')  # B, F, N, P
        res = self.encoder(res)  # B, F, N, d

        # st_conv
        res = self.st_conv(res, adj)  # B, F, N, d

        # predict
        res = rearrange(res, 'b f n d -> b n d f')
        res = self.predict(res)  # B, n, d, S

        # decoder
        res = rearrange(res, 'b n d s -> b s n d')  # B, S, N, d
        res = self.decoder(res)  # B, S, N, seg_len
        res = rearrange(res, 'b s n p -> b (s p) n')  # B, H, N
        res = res[:, :self.horizon, :]

        return res


class LFEncDec(nn.Module):
    def __init__(self,
                 seq_len,
                 horizon,
                 num_nodes,
                 d_embed=64,
                 num_layers=3,
                 dropout=0.05):
        super().__init__()

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes

        self.encoder = nn.Linear(seq_len, d_embed)
        self.cross_var = DirectedGraphConvEnc(d_embed, d_embed, d_embed, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.Linear(d_embed, horizon)

    def forward(self, x, adj):
        B, L, N = x.shape  # L = seq_len

        # Encoding
        res = rearrange(x, 'b l n -> b n l')
        res = self.encoder(res)  # B, N, D

        # cross var
        res = self.cross_var(res, adj)  # B, N, D

        # decoder
        res = self.decoder(res)  # B, N, H
        res = rearrange(res, 'b n h -> b h n')  # B, H, N

        return res


class Model(nn.Module):
    def __init__(self, seq_len, horizon, num_nodes, periods, device,
                 norm_type='global', d_node_embed=32, tanh_alpha=3, num_neighs=3,
                 d_lf_embed=128, d_hf_embed=64, num_layers=0, dropout=0.05):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.norm_type = norm_type

        self.gc = CausalGraphGenerator(num_nodes, d_node_embed, device, alpha=tanh_alpha, num_neighs=num_neighs)

        self.diff_block = LFEncDec(seq_len - 1, horizon, num_nodes, d_embed=d_lf_embed,
                                   num_layers=num_layers, dropout=dropout)

        self.hf_st_conv = STConvEnc(d_hf_embed, d_hf_embed, d_hf_embed, num_layers=num_layers, dropout=dropout)
        self.hf_encoders = nn.ModuleList([
            HFSegEncDec(seq_len, horizon, num_nodes, st_conv=self.hf_st_conv, seg_len=period,
                        d_embed=d_hf_embed) for period in periods
        ])

    @staticmethod
    def batch_normalization(x):
        # x (B, T, N, ...)
        mean = x.mean(dim=(0, 1), keepdim=True).detach()
        std = (x.std(dim=(0, 1), keepdim=True) + 1e-5).detach()
        x = (x - mean) / std
        return x, mean, std

    @staticmethod
    def normalization(x, mean, std):
        return (x - mean) / std

    @staticmethod
    def de_normalization(x, mean, std):
        return x * std + mean

    @staticmethod
    def instance_normalization(x):
        # x (B, T, N, ...)
        mean = x.mean(dim=1, keepdim=True).detach()
        std = (x.std(dim=1, keepdim=True) + 1e-5).detach()
        x = (x - mean) / std
        return x, mean, std

    def forward(self, xl, xh, stats_xl_diff=None, stats_xh=None):
        """
        :param xl: (B, T, N)
        :param xh: (B, T, N, dwt_level)
        :param stats_xl_diff: (2[mean, std], 1, 1, N) global stats
        :param stats_xh: (2[mean, std], 1, 1, N, dwt_level) global stats
        :return:
        """
        # adaptive graph learning
        adp = self.gc(None)

        # region Low-frequency Differencing Forecasting
        # normalization
        xl_diff = diff_1st(xl)
        if self.norm_type == 'global':
            assert stats_xl_diff is not None
            mean_xl_diff, std_xl_diff = stats_xl_diff
            xl_diff = self.normalization(xl_diff, mean_xl_diff, std_xl_diff)
        elif self.norm_type == 'inst':
            xl_diff, mean_xl_diff, std_xl_diff = self.instance_normalization(xl_diff)
        elif self.norm_type == 'batch':
            xl_diff, mean_xl_diff, std_xl_diff = self.batch_normalization(xl_diff)
        else:
            raise ValueError(f'Unknown norm_type: {self.norm_type}')

        # processing
        pred_xl_diff = self.diff_block(xl_diff, adp)  # (B, horizon, N)

        # de-normalization
        pred_xl_diff = self.de_normalization(pred_xl_diff, mean_xl_diff, std_xl_diff)

        # recover prediction from first order diff seq
        pred_xl_diff_cumSum = pred_xl_diff.cumsum(dim=1)  # (B, horizon, N)
        pred_xl = xl[:, -1:, :] + pred_xl_diff_cumSum
        # endregion

        # region High-frequency Forecasting
        # normalization
        if self.norm_type == 'global':
            assert stats_xh is not None
            mean_xh, std_xh = stats_xh
            xh = self.normalization(xh, mean_xh, std_xh)
        elif self.norm_type == 'inst':
            xh, mean_xh, std_xh = self.instance_normalization(xh)
        elif self.norm_type == 'batch':
            xh, mean_xh, std_xh = self.batch_normalization(xh)
        else:
            raise ValueError(f'Unknown norm_type: {self.norm_type}')

        # processing
        dwt_level = xh.shape[-1]
        pred_xh = []
        for i in range(dwt_level):
            pred_xh.append(self.hf_encoders[i](xh[..., i], adp))  # (B, horizon, N)
        pred_xh = torch.stack(pred_xh, dim=-1)  # (B, horizon, N, dwt_level)

        # de-normalization
        pred_xh = self.de_normalization(pred_xh, mean_xh, std_xh)
        # endregion

        # Fusion
        pred_x = pred_xl + pred_xh.sum(-1)

        return pred_x, pred_xl, pred_xh
