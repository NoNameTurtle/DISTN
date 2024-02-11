"""
Disentangled Integrated Network (DISTN)
- Adapted from DISTN_SegV2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nns.model.iTransformer import Model as iTransformer
from nns.model.Crossformer import Model as Crossformer
from nns.model.Linear import Model as Linear
from nns.layers.distn_layers import diff_1st

encoder_dict = {
    'iTransformer': iTransformer,
    'Crossformer': Crossformer,
    'Linear': Linear
}


class Model(nn.Module):
    def __init__(self, seq_len, horizon, dwt_level,
                 original=False, diff_only=False, dwt_only=False,
                 merge_hf=True, independent=False, norm_type='global',
                 encoder='iTransformer', **encoder_kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.original = original
        self.diff_only = diff_only
        self.dwt_only = dwt_only
        self.merge_hf = merge_hf
        self.independent = independent
        self.norm_type = norm_type
        self.dwt_level = dwt_level
        encoder_model = encoder_dict[encoder]

        if self.original:
            self.enc = encoder_model(seq_len, horizon, **encoder_kwargs)
        elif self.diff_only:  # only with diff
            self.diff_enc = encoder_model(seq_len-1, horizon, **encoder_kwargs)
        else:  # with decomposition
            if self.merge_hf:  # merged hf
                self.hf_merged_enc = encoder_model(seq_len, horizon, **encoder_kwargs)
            else:
                if self.independent:  # separate and independent
                    self.hf_enc = nn.ModuleList([
                        encoder_model(seq_len, horizon, **encoder_kwargs) for _ in range(dwt_level)
                    ])
                else:  # separate and shared
                    self.hf_enc = encoder_model(seq_len, horizon, **encoder_kwargs)

            if dwt_only:  # only with decomposition
                self.lf_enc = encoder_model(seq_len, horizon, **encoder_kwargs)
            else:  # with decomposition and diff
                self.lf_diff_enc = encoder_model(seq_len-1, horizon, **encoder_kwargs)

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

    def forward(self, xl, xh, x_mark, x_dec_mark, stats_xl_diff=None, stats_xl=None, stats_xh=None, stats_x_diff=None):
        """
        :param xl: (B, T, N)
        :param xh: (B, T, N, dwt_level)
        :param x_mark: (B, T, D)
        :param x_dec_mark: (B, T, D)
        :param stats_xl_diff: (2[mean, std], 1, 1, N) global stats
        :param stats_xl: (2[mean, std], 1, 1, N) global stats
        :param stats_xh: (2[mean, std], 1, 1, N, dwt_level) global stats
        :param stats_x_diff: (2[mean, std], 1, 1, N) global stats
        :return:
        """
        if self.original:
            x = xl + xh.sum(dim=-1)
            pred_x = self.enc(x, x_mark, None, x_dec_mark)
            return pred_x, None, None
        elif self.diff_only:  # only with diff
            x = xl + xh.sum(dim=-1)
            x_diff = diff_1st(x)
            if self.norm_type == 'global':
                assert stats_x_diff is not None
                mean_x_diff, std_x_diff = stats_x_diff
                x_diff = self.normalization(x_diff, mean_x_diff, std_x_diff)
            elif self.norm_type == 'inst':
                x_diff, mean_x_diff, std_x_diff = self.instance_normalization(x_diff)
            elif self.norm_type == 'batch':
                x_diff, mean_x_diff, std_x_diff = self.batch_normalization(x_diff)
            elif self.norm_type == 'none':
                pass
            else:
                raise ValueError(f'Unknown norm_type: {self.norm_type}')
            pred_x_diff = self.diff_enc(x_diff, x_mark[:, 1:, :], None, x_dec_mark)
            if self.norm_type != 'none':
                pred_x_diff = self.de_normalization(pred_x_diff, mean_x_diff, std_x_diff)
            pred_x = x[:, -1:, :] + pred_x_diff.cumsum(dim=1)
            return pred_x, None, None
        else:
            # region high-frequency forecasting
            if self.merge_hf:
                xh_merged = xh.sum(-1)
                if self.norm_type == 'global':
                    assert stats_xh is not None
                    mean_xh_merged, std_xh_merged = stats_xh.mean(-1)
                    xh_merged = self.normalization(xh_merged, mean_xh_merged, std_xh_merged)
                elif self.norm_type == 'inst':
                    xh_merged, mean_xh_merged, std_xh_merged = self.instance_normalization(xh_merged)
                elif self.norm_type == 'batch':
                    xh_merged, mean_xh_merged, std_xh_merged = self.batch_normalization(xh_merged)
                elif self.norm_type == 'none':
                    pass
                else:
                    raise ValueError(f'Unknown norm_type: {self.norm_type}')
                pred_xh_merged = self.hf_merged_enc(xh_merged, x_mark, None, x_dec_mark)
                if self.norm_type != 'none':
                    pred_xh_merged = self.de_normalization(pred_xh_merged, mean_xh_merged, std_xh_merged)
                pred_xh = pred_xh_merged
            else:
                if self.norm_type == 'global':
                    assert stats_xh is not None
                    mean_xh, std_xh = stats_xh
                    xh = self.normalization(xh, mean_xh, std_xh)
                elif self.norm_type == 'inst':
                    xh, mean_xh, std_xh = self.instance_normalization(xh)
                elif self.norm_type == 'batch':
                    xh, mean_xh, std_xh = self.batch_normalization(xh)
                elif self.norm_type == 'none':
                    pass
                else:
                    raise ValueError(f'Unknown norm_type: {self.norm_type}')
                if self.independent:
                    pred_xh = torch.stack([self.hf_enc[i](xh[..., i], x_mark, None, x_dec_mark) for i in range(self.dwt_level)], dim=-1)
                else:
                    pred_xh = torch.stack([self.hf_enc(xh[..., i], x_mark, None, x_dec_mark) for i in range(self.dwt_level)], dim=-1)
                if self.norm_type != 'none':
                    pred_xh = self.de_normalization(pred_xh, mean_xh, std_xh)
                pred_xh_merged = pred_xh.sum(-1)
            # endregion

            if self.dwt_only:  # only with decomposition
                if self.norm_type == 'global':
                    assert stats_xl is not None
                    mean_xl, std_xl = stats_xl
                    xl = self.normalization(xl, mean_xl, std_xl)
                elif self.norm_type == 'inst':
                    xl, mean_xl, std_xl = self.instance_normalization(xl)
                elif self.norm_type == 'batch':
                    xl, mean_xl, std_xl = self.batch_normalization(xl)
                elif self.norm_type == 'none':
                    pass
                else:
                    raise ValueError(f'Unknown norm_type: {self.norm_type}')
                # xl, mean_xl, std_xl = self.instance_normalization(xl)  # RevIN for non-stationary trends
                pred_xl = self.lf_enc(xl, x_mark, None, x_dec_mark)
                if self.norm_type != 'none':
                    pred_xl = self.de_normalization(pred_xl, mean_xl, std_xl)
            else:  # with decomposition and diff
                xl_diff = diff_1st(xl)
                if self.norm_type == 'global':
                    assert stats_xl_diff is not None
                    mean_xl_diff, std_xl_diff = stats_xl_diff
                    xl_diff = self.normalization(xl_diff, mean_xl_diff, std_xl_diff)
                elif self.norm_type == 'inst':
                    xl_diff, mean_xl_diff, std_xl_diff = self.instance_normalization(xl_diff)
                elif self.norm_type == 'batch':
                    xl_diff, mean_xl_diff, std_xl_diff = self.batch_normalization(xl_diff)
                elif self.norm_type == 'none':
                    pass
                else:
                    raise ValueError(f'Unknown norm_type: {self.norm_type}')
                pred_xl_diff = self.lf_diff_enc(xl_diff, x_mark[:, 1:, :], None, x_dec_mark)
                if self.norm_type != 'none':
                    pred_xl_diff = self.de_normalization(pred_xl_diff, mean_xl_diff, std_xl_diff)
                pred_xl = xl[:, -1:, :] + pred_xl_diff.cumsum(dim=1)

            pred_x = pred_xl + pred_xh_merged
            return pred_x, pred_xl, pred_xh
