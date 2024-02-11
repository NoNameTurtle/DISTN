import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, reduce

from nns.layers.conv_blocks import Temporal_Inception, Mixprop


# region utils functions
def fft_find_period(x, ratio=0.2):
    """
    :param x: (B, T, N)
    :param ratio:
    :return:
    """
    amps_fft = abs(torch.fft.rfft(x, dim=1)).mean(dim=(0, 2))  # (F)
    len_freq = amps_fft.shape[-1]
    k = int(len_freq * ratio)
    amps_fft[0] = 0
    top_freq = amps_fft.topk(k).indices
    top_freq = top_freq.detach().cpu().numpy().mean()
    period = x.shape[1] // top_freq
    return int(period)


def diff_1st(x):
    """
    :param x: (B, T, N)
    :return:
    """
    return x[:, 1:, :] - x[:, :-1, :]
# endregion


# region utils blocks
class MLP(nn.Module):
    """
    Multilayer perceptron to encode/decode high dimension representation of sequential data
    """

    def __init__(self,
                 f_in,
                 f_out,
                 hidden_dim=128,
                 hidden_layers=2,
                 dropout=0.05,
                 activation='tanh'):
        super(MLP, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        if hidden_layers == 0:
            layers = [nn.Identity()]
        elif hidden_layers == 1:
            layers = [nn.Linear(self.f_in, self.f_out)]
        else:
            layers = [nn.Linear(self.f_in, self.hidden_dim),
                      self.activation, nn.Dropout(self.dropout)]
            for i in range(self.hidden_layers - 2):
                layers += [nn.Linear(self.hidden_dim, self.hidden_dim),
                           self.activation, nn.Dropout(dropout)]
            layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x:  B x S x f_in
        # y:  B x S x f_out
        y = self.layers(x)
        return y


class CausalGraphGenerator(nn.Module):
    """
    causal graph learning layer
    """

    def __init__(self,
                 num_nodes,
                 d_embed,
                 device,
                 alpha=3,
                 num_neighs=12):
        super(CausalGraphGenerator, self).__init__()
        self.num_nodes = num_nodes
        self.num_neighs = num_neighs

        self.emb1 = nn.Embedding(num_nodes, d_embed)
        self.emb2 = nn.Embedding(num_nodes, d_embed)
        self.lin1 = nn.Linear(d_embed, d_embed)
        self.lin2 = nn.Linear(d_embed, d_embed)

        self.device = device
        self.alpha = alpha

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, causal_mask=None):
        # binary_adj: (N, N)
        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.gelu(torch.tanh(self.alpha * a))

        # specified mask
        if causal_mask is not None:
            mask = causal_mask.float().to(self.device)
        else:
            mask = torch.zeros(self.idx.size(0), self.idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(self.num_neighs, 1)
            mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self):
        # return graph without mask
        # binary_adj: (N, N)
        nodevec1 = self.emb1(self.idx)
        nodevec2 = self.emb2(self.idx)

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.gelu(torch.tanh(self.alpha * a))
        return adj


class DirectedGraphConv(nn.Module):
    def __init__(self, c_in, c_out, depth=2, alpha=0.05, dropout=0.05):
        super().__init__()
        self.g_conv1 = Mixprop(c_in, c_out, depth=depth, alpha=alpha)
        self.g_conv2 = Mixprop(c_in, c_out, depth=depth, alpha=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x (..., N, D)
        res = self.g_conv1(x, adj) + self.g_conv2(x, adj.transpose(1, 0))

        # output
        res = self.dropout(res + x)
        return res


class DirectedGraphConvEnc(nn.Module):
    def __init__(self, c_in, c_out, hidden_dim, num_layers=2, depth=2, alpha=0.05, dropout=0.05):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            layers = [DirectedGraphConv(c_in, c_out, depth, alpha, dropout)]
        else:
            layers = [DirectedGraphConv(c_in, hidden_dim, depth, alpha, dropout)]
            for _ in range(num_layers - 2):
                layers += [DirectedGraphConv(hidden_dim, hidden_dim, depth, alpha, dropout)]
            layers += [DirectedGraphConv(hidden_dim, c_out, depth, alpha, dropout)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        # x (..., N, D)
        for i in range(self.num_layers):
            x = self.layers[i](x, adj)
        return x


class STConv(nn.Module):
    def __init__(self, c_in, c_out, d_conv, num_kernels=6, depth=2, alpha=0.05, dropout=0.05):
        super().__init__()
        self.t_conv = Temporal_Inception(c_in, d_conv, num_kernels=num_kernels)
        self.g_conv1 = Mixprop(d_conv, c_out, depth=depth, alpha=alpha)
        self.g_conv2 = Mixprop(d_conv, c_out, depth=depth, alpha=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        :param x: (B, T, N, D)
        :param adj: (N, N)
        :return:
        """
        res = rearrange(x, 'b t n d -> b d n t')

        # temporal convolution
        res = self.t_conv(res)

        # spatial convolution
        res = rearrange(res, 'b d n t -> b t n d')
        res = self.g_conv1(res, adj) + self.g_conv2(res, adj.transpose(1, 0))

        # output
        res = self.dropout(res + x)
        return res


class STConvEnc(nn.Module):
    def __init__(self, c_in, c_out, d_conv, num_layers=2, num_kernels=6, depth=2, alpha=0.05, dropout=0.05):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            layers = [STConv(c_in, c_out, d_conv, num_kernels, depth, alpha, dropout)]
        else:
            layers = [STConv(c_in, d_conv, d_conv, num_kernels, depth, alpha, dropout)]
            for _ in range(num_layers - 2):
                layers += [STConv(d_conv, d_conv, d_conv, num_kernels, depth, alpha, dropout)]
            layers += [STConv(d_conv, c_out, d_conv, num_kernels, depth, alpha, dropout)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        # x (B, T, N, D)
        for i in range(self.num_layers):
            x = self.layers[i](x, adj)
        return x


class TConv(nn.Module):
    def __init__(self, c_in, c_out, num_kernels=6, dropout=0.05):
        super().__init__()
        self.t_conv = Temporal_Inception(c_in, c_out, num_kernels=num_kernels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: b d n t
        res = self.t_conv(x)
        return self.dropout(res + x)


class TConvEnc(nn.Module):
    def __init__(self, c_in, c_out, d_conv, num_layers=2, num_kernels=6, dropout=0.05):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            layers = [TConv(c_in, c_out, num_kernels, dropout)]
        else:
            layers = [TConv(c_in, d_conv, num_kernels, dropout)]
            for _ in range(num_layers - 2):
                layers += [TConv(d_conv, d_conv, num_kernels, dropout)]
            layers += [TConv(d_conv, c_out, num_kernels, dropout)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x (b, d, n, t)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x


# endregion


# region Time Series Process Modules
class TimeLinear(nn.Module):
    def __init__(self,
                 seq_len,
                 horizon,
                 num_nodes,
                 cross_linear=False):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes

        self.cross_linear = nn.Linear(num_nodes, num_nodes) if cross_linear else None
        self.Linear = nn.Linear(self.seq_len, self.horizon)

    def forward(self, x):
        """
        :param x: (B, L, N)
        :return:
        """
        B, L, N = x.shape  # L = seq_len

        if self.cross_linear is not None:
            x = self.cross_linear(x)

        x = rearrange(x, 'b l n -> b n l')
        x_pred = self.Linear(x)
        x_pred = rearrange(x_pred, 'b n l -> b l n')

        return x_pred


class TimeSegLinearEncDec(nn.Module):
    def __init__(self,
                 seq_len,
                 horizon,
                 num_nodes,
                 seg_len=8,
                 cross_linear=False,
                 d_model=128,
                 dropout=0.05):
        super().__init__()

        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.seg_len = seg_len
        self.freq = math.ceil(self.seq_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.horizon / self.seg_len)  # segment number of output
        self.padding_len = self.seg_len * self.freq - self.seq_len

        self.encoder = MLP(f_in=self.seg_len, f_out=d_model, activation='relu',
                           hidden_dim=d_model, hidden_layers=2, dropout=dropout)
        self.cross_linear = nn.Linear(num_nodes, num_nodes) if cross_linear else None
        self.Linear = nn.Linear(self.freq, self.step)
        self.decoder = MLP(f_in=d_model, f_out=self.seg_len, activation='relu',
                           hidden_dim=d_model, hidden_layers=2, dropout=dropout)

    def forward(self, x_):
        """
        :param x_: (B, L, N)
        :return:
        """
        B, L, N = x_.shape  # L = seq_len

        # slicing
        res = torch.cat((x_[:, L - self.padding_len:, :], x_), dim=1)  # pad the beginning of the sequence
        res = res.chunk(self.freq, dim=1)  # tuple of (B, seg_len, N)
        res = rearrange(torch.stack(res, dim=1), 'b f p n -> b f n p')  # B, F, N, P
        res = self.encoder(res)  # B, F, N, d_conv

        # cross linear
        if self.cross_linear is not None:
            res = rearrange(res, 'b f n d -> b f d n')
            res = self.cross_linear(res)
            res = rearrange(res, 'b f d n -> b f n d')

        # linear
        res = rearrange(res, 'b f n d -> b d n f')  # B, d_conv, N, F
        res = self.Linear(res)  # B, d_conv, N, S

        # decoder
        res = rearrange(res, 'b d n s -> b s n d')  # B, S, N, d_conv
        res = self.decoder(res)  # B, S, N, seg_len
        res = rearrange(res, 'b s n p -> b (s p) n')  # B, H, N
        res = res[:, :self.horizon, :]

        return res


class TrendDiffSegARBlock(nn.Module):
    def __init__(self,
                 diff_len,
                 horizon,
                 num_nodes,
                 seg_len=8,
                 d_conv=64,
                 d_model=128,
                 num_kernels=6,
                 g_conv_depth=2,
                 prop_alpha=0.05,
                 dropout=0.05):
        super().__init__()

        self.diff_len = diff_len
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.seg_len = seg_len
        self.freq = math.ceil(self.diff_len / self.seg_len)  # segment number of input
        self.step = math.ceil(self.horizon / self.seg_len)  # segment number of output
        self.padding_len = self.seg_len * self.freq - self.diff_len
        self.max_kernel_size = 2 * num_kernels - 1

        # TODO: shared encoder and decoder?
        self.encoder = MLP(f_in=self.seg_len, f_out=d_conv, activation='relu',
                           hidden_dim=d_model, hidden_layers=2, dropout=dropout)
        self.STConv = STConv(d_conv, d_conv, d_conv, num_kernels=num_kernels,
                             depth=g_conv_depth, alpha=prop_alpha, dropout=dropout)
        self.decoder = MLP(f_in=d_conv, f_out=self.seg_len, activation='relu',
                           hidden_dim=d_model, hidden_layers=2, dropout=dropout)

    def forward(self, x_diff, adj):
        """
        :param x_diff: (B, L, N)
        :param adj: (N, N)
        :return:
        """
        B, L, N = x_diff.shape  # L = diff_len

        # slicing
        res = torch.cat((x_diff[:, L - self.padding_len:, :], x_diff), dim=1)  # pad the beginning of the sequence
        res = res.chunk(self.freq, dim=1)  # tuple of (B, seg_len, N)
        res = rearrange(torch.stack(res, dim=1), 'b f p n -> b f n p')  # B, F, N, P
        res = self.encoder(res)  # B, F, N, d_conv

        # reconstruction
        x_diff_rec = torch.concat((res[:, :1], self.STConv(res[:, :-1], adj)), dim=1)  # B, F, N, d_conv
        x_diff_rec = self.decoder(x_diff_rec)  # B, F, N, P
        x_diff_rec = rearrange(x_diff_rec, 'b f n p -> b (f p) n')  # B, L, N
        x_diff_rec = x_diff_rec[:, -self.diff_len:, :]  # B, L, N
        # TODO: -self.diff_len: or :self.diff_len?

        # forecasting
        x_diff_pred, cur_diff = [], res[:, -self.max_kernel_size:]
        for i in range(self.step):
            cur_diff_pred = self.STConv(cur_diff, adj)[:, -1:]  # (B, 1, N, d_conv)
            x_diff_pred.append(cur_diff_pred)
            cur_diff = torch.concat((cur_diff[:, 1:], cur_diff_pred), dim=1)
        x_diff_pred = torch.cat(x_diff_pred, dim=1)  # B, S, d_conv
        x_diff_pred = self.decoder(x_diff_pred)
        x_diff_pred = x_diff_pred.reshape(B, self.step, self.seg_len, N)  # B, S, seg_len, N
        x_diff_pred = x_diff_pred.reshape(B, -1, N)[:, :self.horizon, :]  # B, H, N

        return x_diff_rec, x_diff_pred


class TrendDiffSegAREncDec(nn.Module):
    def __init__(self, seq_len, horizon, num_nodes, seg_len=8, d_conv=64,
                 d_model=128, num_kernels=6, g_conv_depth=2,
                 prop_alpha=0.05, dropout=0.05):
        super().__init__()
        self.diff_blocks = nn.ModuleList([
            TrendDiffSegARBlock(
                diff_len=seq_len - 1, horizon=horizon, num_nodes=num_nodes,
                seg_len=seg_len, d_conv=d_conv, d_model=d_model, num_kernels=num_kernels,
                g_conv_depth=g_conv_depth, prop_alpha=prop_alpha, dropout=dropout
            ) for _ in range(self.num_blocks)
        ])

    def forward(self, x_diff, adp):
        res_x_diff, pred_x_diff = x_diff, None  # (B, seq_len-1, N) & (B, horizon, N)
        for i in range(self.num_blocks):
            x_diff_rec, x_diff_pred = self.diff_blocks[i](res_x_diff, adp)
            res_x_diff = res_x_diff - x_diff_rec
            pred_x_diff = pred_x_diff + x_diff_pred if pred_x_diff is not None else x_diff_pred

        return pred_x_diff
# endregion

