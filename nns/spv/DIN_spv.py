import time
import os
import json

from einops import rearrange, repeat
from math import ceil
import time

import matplotlib.pyplot as plt
from nns.model.DIN import Model
from nns.spv.loss import *
from nns.spv.base_spv import BaseSupervisor
import pywt


class DINSupervisor(BaseSupervisor):
    def __init__(self, **kwargs):
        super(DINSupervisor, self).__init__(**kwargs)

        self.loss_type = self._loss_kwargs.get('loss_type', 'mae')
        self.logger.info(
            f"loss info: {self.loss_type}"
        )
        self.seq_len = self._data_kwargs['seq_len']
        self.horizon = self._data_kwargs['pred_len']
        self._get_train_stats()
        if self._data_kwargs.get('show_dwt', True):
            self.plot_dwt_disentangle(self.data['train_data'][0][0][:, 0])  # x[0, :, 0]

    def _get_model(self):
        self._model_kwargs['seq_len'] = self._data_kwargs['seq_len']
        self._model_kwargs['horizon'] = self._data_kwargs['pred_len']
        self._model_kwargs['dwt_level'] = self._data_kwargs['dwt_level']
        model = Model(**self._model_kwargs)
        return model

    def _get_train_periods(self):
        if self._data_kwargs.get('periods', 'none') == 'none':
            self.logger.info('Calculating periods for each level of DWT...')
            dwt_level = int(self._data_kwargs.get('dwt_level', 4))
            freq_ratio = self._model_kwargs.get('freq_ratio', 0.1)
            num_train_sample = len(self.data['train_data'])
            train_x = np.stack([self.data['train_data'][i][0] for i in range(num_train_sample)], axis=0)
            T = train_x.shape[1]
            train_xl, train_xh = self.dwt_disentangle(train_x, to_torch=False)

            periods = []
            for i in range(dwt_level):
                amps_fft = abs(np.fft.rfft(train_xh[..., i], axis=1)).mean(axis=(0, 2))
                len_freq = amps_fft.shape[0]
                top_freq = np.argsort(amps_fft)[-int(len_freq * freq_ratio):].mean()
                periods.append(ceil(T / top_freq))
        else:
            self.logger.info('Using predefined periods...')
            periods = self._data_kwargs.get('periods')
        self.logger.info(f"Train periods: {periods}")
        return periods

    def _get_train_stats(self):
        num_train_sample = len(self.data['train_data'])
        train_x = np.stack([self.data['train_data'][i][0] for i in range(num_train_sample)], axis=0)  # (B, N, T)

        seq_x = np.concatenate([train_x[:, 0, :], train_x[-1, 1:, :]], axis=0)  # (total_len, N)

        seq_x_diff = torch.from_numpy(seq_x[1:, :] - seq_x[:-1, :]).float()
        mean_x_diff = seq_x_diff.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, N)
        std_x_diff = seq_x_diff.std(dim=0, keepdim=True).unsqueeze(0)
        self.stats_x_diff = torch.stack([mean_x_diff, std_x_diff], dim=0).to(self.device)  # (2, 1, N)

        seq_xl, seq_xh = self.dwt_disentangle(seq_x, to_torch=True, axis=0)
        mean_xl = seq_xl.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, N)
        std_xl = seq_xl.std(dim=0, keepdim=True).unsqueeze(0)
        self.stats_xl = torch.stack([mean_xl, std_xl], dim=0).to(self.device)  # (2, 1, 1, N)

        seq_xl_diff = seq_xl[1:, :] - seq_xl[:-1, :]
        mean_xl_diff = seq_xl_diff.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, N)
        std_xl_diff = seq_xl_diff.std(dim=0, keepdim=True).unsqueeze(0)
        self.stats_xl_diff = torch.stack([mean_xl_diff, std_xl_diff], dim=0).to(self.device)  # (2, 1, 1, N)

        mean_xh = seq_xh.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, N, dwt_level)
        std_xh = seq_xh.std(dim=0, keepdim=True).unsqueeze(0)
        self.stats_xh = torch.stack([mean_xh, std_xh], dim=0).to(self.device)  # (2, 1, 1, N, dwt_level)

    def _gen_run_id(self):
        batch_size = self._data_kwargs.get('batch_size')
        wavelet = self._data_kwargs.get('wavelet')
        dwt_level = self._data_kwargs.get('dwt_level')
        model_id = self._kwargs.get('model_id', 'DISTN')

        seq_len = self._data_kwargs.get('seq_len', 48)
        horizon = self._data_kwargs.get('pred_len', 96)

        # region loss settings
        lr_type = self._train_kwargs.get('lr_type', 'StepLR')
        loss_type = self._loss_kwargs.get('loss_type', 'mae')
        step_size = self._train_kwargs.get('step_size', 1)
        lr_milestones = '_'.join([str(_) for _ in self._train_kwargs.get('lr_milestones', [100])])
        lr_decay_ratio = self._train_kwargs.get('lr_decay_ratio', 0.1)
        learning_rate = self._train_kwargs.get('base_lr')
        weight_decay = self._train_kwargs.get('weight_decay', 0.)
        epochs = self._train_kwargs.get('epochs', 10)

        loss_settings = f'_{loss_type}_{lr_type}_lr{learning_rate:.4f}_wd{weight_decay}_ep{epochs}'
        if lr_type == 'StepLR':
            loss_settings += f'_st{step_size}_ldr{lr_decay_ratio}'
        elif lr_type == 'MultiStepLR':
            loss_settings += f'_ms{lr_milestones}_ldr{lr_decay_ratio}'
        else:
            pass
        # endregion

        run_time = self.run_time

        encoder = self._model_kwargs.get('encoder')
        if encoder == 'iTransformer':
            e_layers = self._model_kwargs.get('e_layers', 1)
            d_model = self._model_kwargs.get('d_model', 128)
            d_ff = self._model_kwargs.get('d_ff', 256)
            use_norm = 1 if self._model_kwargs.get('use_norm', False) else 0
            encoder_settings = f'_el{e_layers}_dm{d_model}_df{d_ff}_norm{use_norm}'
        elif encoder == 'Crossformer':
            e_layers = self._model_kwargs.get('e_layers', 1)
            d_model = self._model_kwargs.get('d_model', 128)
            d_ff = self._model_kwargs.get('d_ff', 256)
            encoder_settings = f'_el{e_layers}_dm{d_model}_df{d_ff}'
        else:
            encoder_settings = ''
        original = 1 if self._model_kwargs.get('original', False) else 0
        diff_only = 1 if self._model_kwargs.get('diff_only', False) else 0
        dwt_only = 1 if self._model_kwargs.get('dwt_only', False) else 0
        merge_hf = 1 if self._model_kwargs.get('merge_hf', False) else 0
        independent = 1 if self._model_kwargs.get('independent', False) else 0
        norm_type = self._model_kwargs.get('norm_type', 'global')
        model_settings = f'_o{original}_d{diff_only}_w{dwt_only}_m{merge_hf}_i{independent}_{norm_type}{encoder_settings}'

        data_name = self._data_kwargs['data_name']

        run_id = f'{data_name}_sl{seq_len}_hl{horizon}_{model_id}_bs{batch_size}_{wavelet}_l{dwt_level}' \
                 f'{model_settings}{loss_settings}_{run_time}'

        return run_id

    def loss(self, y_pred, y_true):
        """
        input: y_pred, y_true, yl_pred yl_true (B, T, N)
               yh_pred, yh_true (B, T, N, dwt_level)
        """
        # a, al, ah = self._loss_kwargs.get('a', 1.), self._loss_kwargs.get('al', 0.), self._loss_kwargs.get('ah', 0.)
        # ah_stats = self._loss_kwargs.get('ah_stats', 0.)

        if self.loss_type == 'mae':
            criterion = nn.L1Loss()
        elif self.loss_type == 'mse':
            criterion = nn.MSELoss()
        else:
            raise Exception('Unknown loss type')

        # region series loss calculation
        y_loss = criterion(y_pred, y_true)
        # yl_loss = criterion(yl_pred, yl_true)
        # yh_loss = criterion(yh_pred, yh_true)
        # endregion

        # region stats loss calculation
        # mean_yh = yh_true.mean(dim=1)  # B, N, level
        # std_yh = yh_true.std(dim=1)
        # mean_yh_pred = yh_pred.mean(dim=1)
        # std_yh_pred = yh_pred.std(dim=1)
        # yh_stats_loss = criterion(mean_yh_pred, mean_yh) + criterion(std_yh_pred, std_yh)
        # endregion

        pred_loss = y_loss

        if torch.isnan(pred_loss).item():
            self.logger.info('nan occur in loss computation')

        return pred_loss, {
            "pred_loss": pred_loss,
            "y_loss": y_loss,
            # "yl_loss": yl_loss,
            # "yh_loss": yh_loss,
            # "yh_stats_loss": yh_stats_loss
        }

    def prepare_data(self, x, y, x_mark, y_mark, dwt_y=False):
        """ split x
        input:
            x: (B, seq_len, N)
            y: (B, label_len + horizon, N)
            x_mark, y_mark: (B, seq_len OR horizon, D)
        output:
            x, xl: (B, seq_len, N)
            y, yl: (B, horizon, N)
            xh, yh: (B, seq_len OR horizon, N, dwt_level)
            x_mark, y_mark: (B, seq_len OR horizon, D)
        :return:
        """
        y, y_mark = y[:, -self.horizon:, :], y_mark[:, -self.horizon:, :]
        xl, xh = self.dwt_disentangle(x)
        if dwt_y:
            yl, yh = self.dwt_disentangle(y)
            yl, yh = yl.to(self.device), yh.to(self.device)
        else:
            yl, yh = None, None
        return (
            xl.to(self.device), xh.to(self.device), yl, yh, y.float().to(self.device),
            x_mark.float().to(self.device), y_mark.float().to(self.device)
        )

    def dwt_disentangle(self, x, to_torch=True, axis=1):
        """
        Since the train_loader is shuffled at each epoch, we cannot pre-process the train data with dwt
        :param x: default (B, T, ...)
        :param to_torch: Bool
        :param axis: int
        :return:
        """
        wavelet = self._data_kwargs.get('wavelet', 'coif1')
        dwt_level = int(self._data_kwargs.get('dwt_level', 4))

        coef = pywt.wavedec(x, wavelet=wavelet, level=dwt_level, axis=axis)
        x_rec = []
        for i in range(len(coef)):
            coef_rec = [np.zeros(coef_i.shape) for coef_i in coef]
            coef_rec[i] = coef[i]
            x_rec.append(pywt.waverec(coef_rec, wavelet=wavelet, axis=axis))
        xl = x_rec[0]
        xh = x_rec[1:]

        if to_torch:
            xl = torch.tensor(xl).float()
            xh = torch.stack([torch.tensor(_).float() for _ in xh], dim=-1)
        else:
            xh = np.stack(xh, axis=-1)
        return xl, xh

    def plot_dwt_disentangle(self, x):
        """
        :param x: numpy.array (T)
        :return:
        """
        dwt_level = int(self._data_kwargs.get('dwt_level', 4))
        xl, xh = self.dwt_disentangle(x.reshape(1, *x.shape), to_torch=False)

        num_subplots = dwt_level + 2
        n_row = ceil(num_subplots / 2)
        fig, ax = plt.subplots(n_row, 2, figsize=(12, 3 * n_row))
        ax[0, 0].plot(x)
        ax[0, 0].set_title("original")
        ax[0, 1].plot(xl[0])
        ax[0, 1].set_title("xl")
        for i in range(2, num_subplots):
            row_i = i // 2
            col_i = i % 2
            ax[row_i, col_i].plot(xh[i-2][0])
            ax[row_i, col_i].set_title(f"xh_{dwt_level-(i-2)}")
        fig.tight_layout()
        if self._save_and_log:
            plt.savefig(os.path.join(self.log_dir, 'dwt_disentangle.png'))
        else:
            plt.show()
        plt.close()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_loss_dict = {}
        for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.train_loader):
            self.train_iter += 1

            xl, xh, yl, yh, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)

            self.optimizer.zero_grad()

            if self._train_kwargs.get('debug', False):
                torch.autograd.set_detect_anomaly(True)

            y_pred, _, _ = self.model(xl, xh, x_mark, y_mark, self.stats_xl_diff,
                                      self.stats_xl, self.stats_xh, self.stats_x_diff)
            loss, loss_dict = self.loss(y_pred, y)

            if self._train_kwargs.get('debug', False):
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()

            # add max grad clipping
            if self._train_kwargs.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._train_kwargs['max_grad_norm'])

            self.optimizer.step()
            total_loss += loss.item()
            for k in loss_dict.keys():
                if k in total_loss_dict.keys():
                    total_loss_dict[k] += loss_dict[k].item()
                else:
                    total_loss_dict[k] = loss_dict[k].item()

            # log information
            if batch_idx % self._train_kwargs['log_step'] == 0:
                self.logger.info(f'Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.3f}')\

        train_epoch_loss = total_loss / self.train_per_epoch
        for k in total_loss_dict.keys():
            total_loss_dict[k] = round(total_loss_dict[k] / self.train_per_epoch, 3)
        self.logger.info(f'>>>>>>>>Train Epoch {epoch}: averaged loss {train_epoch_loss:.3f}')
        self.logger.info(f'\tTrain Epoch {epoch}: averaged loss in details {total_loss_dict}')
        return train_epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        total_val_loss_dict = {}
        with torch.no_grad():
            for batch_idx, (x, y, x_mark, y_mark) in enumerate(self.val_loader):
                xl, xh, yl, yh, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)
                y_pred, _, _ = self.model(xl, xh, x_mark, y_mark, self.stats_xl_diff,
                                          self.stats_xl, self.stats_xh, self.stats_x_diff)
                loss, loss_dict = self.loss(y_pred, y)

                total_val_loss += loss.item()
                for k in loss_dict.keys():
                    if k in total_val_loss_dict.keys():
                        total_val_loss_dict[k] += loss_dict[k].item()
                    else:
                        total_val_loss_dict[k] = loss_dict[k].item()
        val_loss = total_val_loss / self.val_per_epoch
        for k in total_val_loss_dict.keys():
            total_val_loss_dict[k] = round(total_val_loss_dict[k] / self.val_per_epoch, 3)
        self.logger.info(f'\tVal Epoch {epoch}: average loss: {val_loss:.6f}')
        self.logger.info(f'\tVal Epoch {epoch}: average loss in details {total_val_loss_dict}')
        return val_loss

    def test(self, test_loader=None, message=None, plot_id=None):
        if test_loader is None:
            test_loader = self.test_loader
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch_idx, (x, y, x_mark, y_mark) in enumerate(test_loader):
                xl, xh, yl, yh, y, x_mark, y_mark = self.prepare_data(x, y, x_mark, y_mark)
                y_pred, _, _ = self.model(xl, xh, x_mark, y_mark, self.stats_xl_diff,
                                          self.stats_xl, self.stats_xh, self.stats_x_diff)

                preds.append(y_pred)
                labels.append(y)

        preds = torch.cat(preds, dim=0).detach().cpu()  # (B, T, N)
        labels = torch.cat(labels, dim=0).detach().cpu()
        print(f"y_preds {preds.shape}; labels {labels.shape}")

        key_hor_list = [0] + np.arange(47, self.horizon, 48).tolist()

        # region loss dict
        mae_overall, mse_overall, _, smape_overall, r2_overall = metrics(preds, labels)
        loss_dict = {
            'mae_overall': mae_overall,
            'mse_overall': mse_overall,
            'smape_overall': smape_overall,
            'r2_overall': r2_overall
        }
        self.logger.info(f"Horizon Average: MSE {loss_dict['mse_overall']:.3f}, MAE {loss_dict['mae_overall']:.3f}")
        for hor in key_hor_list:
            mae_hor, mse_hor, _, smape_hor, r2_hor = metrics(preds[:, hor, :], labels[:, hor, :])
            loss_dict[f'hor_{hor + 1}'] = {
                'mae': mae_hor, 'mse': mse_hor, 'smape': smape_hor, 'r2': r2_hor
            }
        # endregion

        # region plot
        if plot_id is not None and self._save_and_log:
            for hor in key_hor_list:
                cur_loss_dict = {
                    'mae': MAE_torch(preds[:, hor, plot_id], labels[:, hor, plot_id]).item(),
                    'mse': MSE_torch(preds[:, hor, plot_id], labels[:, hor, plot_id]).item(),
                    'smape': SMAPE_torch(preds[:, hor, plot_id], labels[:, hor, plot_id]).item(),
                    'r2': r2_score_torch(preds[:, hor, plot_id].reshape(-1), labels[:, hor, plot_id].reshape(-1))
                }
                self.plot_sequence(labels[:, hor, plot_id], preds[:, hor, plot_id],
                                   f'sequence_var_{plot_id}_hor_{hor + 1}')
                self.plot_hor_ret(labels[:, hor, plot_id], preds[:, hor, plot_id],
                                  cur_loss_dict, f'scatters_var_{plot_id}_hor_{hor + 1}')
        # endregion

        # region save test results
        # quick save
        settings = self._run_id if message is None else self._run_id + '_' + message
        model_id = self._kwargs.get('model_id', 'DISTN')
        data_name = self._data_kwargs['data_name']
        local_flag = False if 'root' in self.log_dir else True
        file_name = f'result_{model_id}_{data_name}.txt' if local_flag else f'result_{model_id}_{data_name}_server.txt'
        if not os.path.exists('rets'):
            os.makedirs('rets')
        with open(os.path.join('rets', file_name), 'a') as f:
            f.write(settings + '\n')
            f.write(f"Horizon Average: MSE {mse_overall:.4f}, MAE {mae_overall:.4f}\n\n")

        # save in log_dir
        if self._save_and_log:
            save_dict = {
                'settings': settings,
                'rets': loss_dict
            }
            with open(os.path.join(self.log_dir, 'test_record.txt'), 'w') as f:
                json.dump(save_dict, f, indent=2)

        # save hyperparameters and test loss to tensorboard
        if self._save_tb:
            hparams_dict = {
                'run_time': self.run_time,
                'wavelet': self._data_kwargs.get('wavelet', 'coif1'),
                'dwt_level': self._data_kwargs.get('dwt_level', 4),
                'loss_type': self.loss_type,
                'lr_type': self._train_kwargs.get('lr_type', 'StepLR'),
                'step_size': self._train_kwargs.get('step_size', 1),
                'lr_decay_ratio': self._train_kwargs.get('lr_decay_ratio', 0.5),
                'learning_rate': self._train_kwargs.get('base_lr')
            }
            hparams_dict.update(self._model_kwargs)
            self._writer.add_hparams(
                hparams_dict,
                {'hparams/MSE': mse_overall, 'hparams/MAE': mae_overall}
            )
            self._writer.close()
        # endregion

        return {
            'test_data': (labels, preds),
            'loss': loss_dict
        }

    def plot_hor_ret(self, label, pred, cur_loss_dict, fig_name):
        """
        :param label: (B,)
        :param pred: (B,)
        :param cur_loss_dict:
        :param fig_name:
        :return:
        """
        mae, mse, smape, r2 = (
            cur_loss_dict['mae'], cur_loss_dict['mse'],  cur_loss_dict['smape'], cur_loss_dict['r2']
        )

        fig, ax = plt.subplots(2, 1, figsize=(16, 12))
        max_y = min(label.max(), pred.max())
        ax[0].scatter(label, pred)
        ax[0].plot((0, max_y), (0, max_y), linestyle='-.', color='red')
        ax[0].set_xlabel('ground truth')
        ax[0].set_ylabel('pred')
        ax[0].set_title(f'true v.s. prediction (r2 {r2:.3f}, mae {mae: .3f}, '
                        f'mse {mse:.3f}, smape {smape * 100:.2f}%)')

        ax[1].scatter(np.arange(label.shape[0]), label - pred)
        ax[1].axhline(y=0, linestyle='-.', color='red')
        ax[1].set_ylabel('residual')
        fig.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'{fig_name}.png'))
        plt.close()

    def plot_sequence(self, label, pred, fig_name):
        # label, pred (T,)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(label, label='ground truth')
        ax.plot(pred, label='pred')
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'{fig_name}.png'))
        plt.close()
