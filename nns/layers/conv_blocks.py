import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x (B, C, H, W)
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Temporal_Inception(nn.Module):
    """
    Causal Inception Layer for Time Series
    - Causal convolution with padding before the first element
    """
    def __init__(self, c_in, c_out, num_kernels=6):
        super(Temporal_Inception, self).__init__()
        self.num_kernels = num_kernels
        conv_blocks = []
        for i in range(num_kernels):
            conv_blocks.append(nn.Conv2d(c_in, c_out, kernel_size=(1, 2 * i + 1)))
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x):
        # input: (B, D, N, T)
        res_list = []
        for i in range(self.num_kernels):
            left_pad = 2 * i
            x_pad = F.pad(x, (left_pad, 0, 0, 0))  # causal padding with zeros
            res_list.append(self.conv_blocks[i](x_pad))  # B, D, N, T
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Mixprop(nn.Module):
    """
    mix-hop propagation layer (same as MTGNN)
    """

    def __init__(self, c_in, c_out, depth, alpha):
        super(Mixprop, self).__init__()
        self.depth = depth
        self.alpha = alpha

        self.fuse = nn.Linear((depth + 1) * c_in, c_out)

    @staticmethod
    def spatial_graph_norm(adj):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        d = adj.sum(dim=1)
        return adj / d.view(-1, 1)

    @staticmethod
    def oneStepProp(x, A):
        """
        one-step propagation along graph A
        :param x: (..., N, D)
        :param A: a directed adjacency matrix of shape (N, N), a_ij represents the edge from i to j
        :return:
        """
        x = torch.einsum('...vd, vw->...wd', (x, A))
        return x.contiguous()

    def forward(self, x, adj):
        """
        :param x: (..., N, D);
        :param adj: a directed adjacency matrix of shape (N, N), a_ij represents the edge from i to j
        :return: (..., N, c_out)
        """
        h = x
        out = [h]
        adj_normed = self.spatial_graph_norm(adj)
        for i in range(self.depth):
            h = self.alpha * x + (1 - self.alpha) * self.oneStepProp(h, adj_normed)  # (..., N, D)
            out.append(h)
        ho = torch.cat(out, dim=-1)  # (..., N, (depth+1)*D)
        ho = self.fuse(ho)
        return ho


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res