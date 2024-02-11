import argparse
import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm

from nns.spv.DISTN_spv import DISTNSupervisor
from utils.path_utils import load_spv_config
from utils.train_inits import init_seed

parser = argparse.ArgumentParser(description='DISTN for MTS Forecasting')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--dataset', type=str, required=False, default='ETTh2', help='dataset name')


args = parser.parse_args()
fix_seed = args.seed
dataset = args.dataset


def train_kdd(model_id, itr=1):
    horizon_data_settings = {
        96: {
            'dwt_level': 3
        },
        192: {
            'dwt_level': 3
        },
        336: {
            'dwt_level': 3
        },
        720: {
            'dwt_level': 2
        }
    }
    horizon_model_settings = {
        96: {
            'd_lf_embed': 256,
            'd_hf_embed': 64,
            'num_layers': 1,
        },
        192: {
            'd_lf_embed': 128,
            'd_hf_embed': 32,
            'num_layers': 1,
        },
        336: {
            'd_lf_embed': 128,
            'd_hf_embed': 64,
            'num_layers': 1,
        },
        720: {
            'd_lf_embed': 512,
            'd_hf_embed': 128,
            'num_layers': 1,
        }
    }

    n_total_trial = len(horizon_model_settings) * itr
    pbar = tqdm(total=n_total_trial, desc=f'Training {model_id} with KDD24')
    rep_rets = {
        'horizon': [],
        'itr': [],
        'mse': [],
        'mae': []
    }

    for horizon in [96, 192, 336, 720]:
        print('---------------------------------------------')
        print(f'\tpred_len: {horizon}, seed: {fix_seed}')
        print('---------------------------------------------')
        init_seed(fix_seed)
        mae_list, mse_list = [], []
        for ii in range(itr):
            print(f'>>>>>>>>>>>>Iteration {ii + 1}/{itr}')
            spv_config = load_spv_config(model_id, dataset=dataset)
            spv_config['data']['pred_len'] = horizon
            spv_config['data'].update(horizon_data_settings[horizon])
            spv_config['model'].update(horizon_model_settings[horizon])

            spv = DISTNSupervisor(**spv_config)
            spv.train()
            loss = spv.test(message=f"s{fix_seed}_{ii}", plot_id=None)['loss']
            mae_list.append(loss['mae_overall'])
            mse_list.append(loss['mse_overall'])

            # update rep_rets
            rep_rets['horizon'].append(horizon)
            rep_rets['itr'].append(ii)
            rep_rets['mse'].append(loss['mse_overall'])
            rep_rets['mae'].append(loss['mae_overall'])

            pbar.update()
        print(f'>>>>>>>>>>>>MAE: {np.mean(mae_list):.4f}, MSE: {np.mean(mse_list):.4f}')

    # save results to csv
    rep_rets = pd.DataFrame(rep_rets)
    if not os.path.exists('rets'):
        os.makedirs('rets')
    rep_rets.to_csv(f'rets/{model_id}_kdd24_{dataset}_s{fix_seed}.csv', index=False)
    print("Train finished")


if __name__ == "__main__":
    train_kdd('DISTN', itr=args.itr)
