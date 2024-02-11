"""
Update history
* 2023/0921
    - add support to optuna: set optuna trial as a input to spv.train(
"""

import os
import yaml
import time
import math
import torch
import optuna
import json
import matplotlib.pyplot as plt
import copy
from torch.utils.tensorboard import SummaryWriter
from abc import ABCMeta, abstractmethod
from tqdm import tqdm

from utils.data_utils import data_provider
from utils.logger import get_logger

import time

DEBUG_MODE = True

""" Supervisor config list
log_dir: data/model_logs/***
log_level: DEBUG
save_and_log: false

data:
  batch_size: 64
  dataset_dir: data/METR-LA-***

model:
  description: ***
  special_id: ***
  *model specific*: ***

loss:
  *loss specific*: ***

train:
  epoch: 0
  epochs: 100
  log_step: 10
  tensorboard_dir: runs/***
  plot: true
  optimizer: adam
  epsilon: 1.0e-3
  lr_type: MultiStepLR
  base_lr: 0.001
  lr_milestones: [20, 30, 40, 50]
  lr_decay_ratio: 0.1
  max_grad_norm: 5
  early_stop: 10
"""


def load_checkpoint(log_dir, Supervisor, save_and_log=False, save_tb=False,
                    load_data_flag=True, batch_size=64, **kwargs):
    check_point = torch.load(os.path.join(log_dir, 'best_model.pth'))
    state_dict = check_point['state_dict']
    with open(os.path.join(log_dir, 'config.txt'), 'r') as f:
        saved_kwargs = json.load(f)

    saved_kwargs['pre_trained'] = True
    saved_kwargs['log_dir'] = log_dir  # set to current log_dir
    saved_kwargs['save_and_log'] = save_and_log
    saved_kwargs['save_tb'] = save_tb
    saved_kwargs['load_data_flag'] = load_data_flag
    if load_data_flag and batch_size is not None:
        saved_kwargs['data']['batch_size'] = batch_size
    for key, value in kwargs.items():  # additional params to changed
        saved_kwargs[key] = value

    supervisor = Supervisor(**saved_kwargs)
    supervisor.model.load_state_dict(state_dict)
    return supervisor


class BaseSupervisor(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._loss_kwargs = kwargs.get('loss')
        self.device = torch.device(kwargs.get('device', 'cuda:0'))

        # supervisor state
        self._save_and_log = self._kwargs.get('save_and_log', True)
        self._save_tb = self._kwargs.get('save_tb', True)

        # logging
        self.run_time = time.strftime('%m%d%H%M%S')
        self._run_id = self._gen_run_id()
        log_level = self._kwargs.get('log_level', 'INFO')
        if self._kwargs.get('pre_trained', False):
            self.log_dir = self._get_log_dir(kwargs, None)  # use log_dir directly
        else:
            self.log_dir = self._get_log_dir(kwargs, self._run_id)
        if self._save_and_log:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.logger = get_logger(self.log_dir, __name__, 'info.log', level=log_level, write_file=self._save_and_log)
        self.logger.info(f"Run ID: {self._run_id}")
        if self._save_tb:
            tb_subdir = self._kwargs.get('tb_subdir', 'runs')
            self.tb_dir = os.path.join(self._kwargs.get('log_dir', '.'), tb_subdir, self._run_id)
            self._writer = SummaryWriter(self.tb_dir)

        # data set
        if self._kwargs.get('load_data_flag', True):
            self.data = self.load_dataset()
            self.train_loader = self.data['train_loader']
            self.val_loader = self.data['val_loader']
            self.test_loader = self.data['test_loader']
            self.train_per_epoch = len(self.train_loader)
            self.val_per_epoch = len(self.val_loader)
            self.logger.info(f"Sample size: train {len(self.data['train_data'])}, "
                             f"val {len(self.data['val_data'])}, "
                             f"test {len(self.data['test_data'])}")

        # model
        self.model = self._get_model()
        self.model = self.model.to(self.device)
        self.logger.info("Model created")
        self.logger.info(f"Model Desc: {self._kwargs['description']}")

        # training
        self.train_iter = 0  # starts from 1 when training
        self.task_level = 1  # for cl of multistep prediction, level means horizon
        self.cl_learn = self._train_kwargs.get('cl_learn', False)
        if self.cl_learn:
            self.cl_step = self._train_kwargs.get('cl_step', 100)
            self.logger.info(f"Train model with curriculum learning, step {self.cl_step}")
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.optimizer = None
        self.lr_scheduler = None
        self.reset_optimizer_and_scheduler()

        # model saving
        self._save_model_dir = self.log_dir
        self.save_config()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self._data_kwargs, flag)
        return data_set, data_loader

    def load_dataset(self):
        # traditional data which has the format of (data, target) for each sample
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        data = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }
        return data

    def load_checkpoint(self, log_dir):
        with open(os.path.join(log_dir, 'config.txt'), 'r') as f:
            model_params = json.load(f)['model']
        # check hyperparams
        # for key, value in self._model_kwargs:
        #     assert value == model_params[key]
        check_point = torch.load(os.path.join(log_dir, 'best_model.pth'))
        state_dict = check_point['state_dict']
        self.model.load_state_dict(state_dict)
        self.logger.info(f'load checkpoint from {log_dir} successfully')

    def reset_optimizer_and_scheduler(self):
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self._train_kwargs.get('base_lr', 0.001),
                                          weight_decay=self._train_kwargs.get('weight_decay', 0))

        # lr scheduler
        if self._train_kwargs.get('lr_type', 'MultiStepLR') == 'MultiStepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                     milestones=self._train_kwargs.get('lr_milestones', [20, 30, 40, 50]),
                                                                     gamma=self._train_kwargs.get('lr_decay_ratio', 0.1))
        elif self._train_kwargs.get('lr_type') == 'StepLR':
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                                step_size=int(self._train_kwargs.get('step_size', 1)),
                                                                gamma=self._train_kwargs.get('lr_decay_ratio', 0.1))
        else:
            raise Exception("Unknown lr_type")

    def save_checkpoint(self, epoch):
        if not self._save_and_log:
            self.logger.info("Save_and log is false. No model is saved")
            return

        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self._kwargs
        }
        torch.save(state, self._save_model_dir + f'/best_model.pth')
        self.logger.info("Saved current best model at {}".format(epoch))

    def save_config(self):
        if self._save_and_log:
            with open(os.path.join(self.log_dir, 'config.txt'), 'w') as f:
                json.dump(self._kwargs, f, indent=2)

    def train(self, trial=None):
        # trial is used for optuna when hyperparameters optimization is needed
        return self._train(trial)

    def _train(self, trial=None):
        self.train_iter = 0
        if self._train_kwargs.get('debug', False):
            self.logger.info("Debug Mode")
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []

        # print_model_parameters(self.model)
        start_time = time.time()
        for epoch in tqdm(range(1, self._train_kwargs['epochs'] + 1)):
            train_epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            epoch_cost_time = time.time() - train_epoch_time
            self.logger.info(f"\tTrain Epoch {epoch} cost time: {epoch_cost_time:.4f}s; "
                             f"Train Speed {epoch_cost_time / self.train_per_epoch:.4f}s/iter")
            if math.isnan(train_epoch_loss):
                self.logger('Found nan loss, training loop break!')
                break
            # learning rate decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.val_loader is None:
                val_epoch_loss = 0
            else:
                val_epoch_loss = self.val_epoch(epoch)
            self.logger.info(f"\tUpdating LR to: {[group['lr'] for group in self.optimizer.param_groups]}")
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            if self.val_loader is None:
                val_epoch_loss = train_epoch_loss

            # save tensorboard
            if self._save_tb:
                self._writer.add_scalar('loss/train_loss', train_epoch_loss, epoch)
                self._writer.add_scalar('loss/val_loss', val_epoch_loss, epoch)

            # early stop
            if val_epoch_loss < best_loss - 1e-5:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                if epoch < self._train_kwargs['min_epochs']:
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                    self.logger.info(f"\tnot_improved_count: {not_improved_count}")
                best_state = False

            if trial is not None:
                trial.report(val_epoch_loss, epoch)
                if trial.should_prune():  # early stop by optuna (therefore no need to remember the best_loss)
                    raise optuna.exceptions.TrialPruned()

            # user-defined early stop
            if not_improved_count == self._train_kwargs['early_stop']:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self._train_kwargs['early_stop']))
                break

            # save the best state
            if best_state:
                self.save_checkpoint(epoch)
                best_model = copy.deepcopy(self.model.state_dict())
            # plot loss figure
            if self._train_kwargs['plot_loss'] and self._save_and_log:
                self._plot_line_figure([train_loss_list, val_loss_list], path=self._save_model_dir,
                                       warmup=self._train_kwargs.get('plot_warmup', 5))
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format(training_time / 60))
        self.model.load_state_dict(best_model)
        if self._save_tb:
            self._writer.flush()
        return best_loss

    @staticmethod
    def _get_log_dir(kwargs, run_id=None):
        log_dir = kwargs.get('log_dir')
        if log_dir is None:
            log_dir = os.getcwd()
        if run_id is not None:
            log_dir = os.path.join(log_dir, run_id)
        else:
            log_dir = os.path.join(log_dir)

        return log_dir

    @staticmethod
    def _plot_line_figure(losses, path, warmup=5):
        # whole loss
        train_loss = losses[0]
        val_loss = losses[1]
        plt.style.use('ggplot')
        epochs = list(range(1, len(train_loss) + 1))
        plt.plot(epochs, train_loss, 'r-o')
        plt.plot(epochs, val_loss, 'b-o')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(os.path.join(path, 'loss.png'), bbox_inches="tight")
        plt.cla()
        plt.close("all")

        # loss with warmup
        train_loss = losses[0][(warmup - 1):]
        val_loss = losses[1][(warmup - 1):]
        plt.style.use('ggplot')
        epochs = list(range(1, len(train_loss) + 1))
        plt.plot(epochs, train_loss, 'r-o')
        plt.plot(epochs, val_loss, 'b-o')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(os.path.join(path, 'loss_warmup.png'), bbox_inches="tight")
        plt.cla()
        plt.close("all")

    @abstractmethod
    def _gen_run_id(self):
        pass

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def val_epoch(self, epoch):
        pass

    def get_run_id(self):
        return self._run_id

    def get_run_time(self):
        return self.run_time
