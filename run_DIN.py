import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

from nns.spv.DIN_spv import DINSupervisor as Supervisor
from utils.path_utils import load_spv_config
from utils.train_inits import init_seed

parser = argparse.ArgumentParser(description='Disentangled Integrated Framework for MTS Forecasting')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--backbone', type=str, required=False, default='iTransformer', help='backbone model id for DIN')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--dataset', type=str, required=False, default='ETTh2', help='dataset name')


args = parser.parse_args()
fix_seed = args.seed
dataset = args.dataset


def train_kdd(model_id, itr=1, save_id=''):
    pred_len_list = [96, 192, 336, 720]
    n_total_trial = len(pred_len_list) * itr
    pbar = tqdm(total=n_total_trial, desc=f'Training {model_id} with KDD')
    rep_rets = {
        'run_id': [],
        'horizon': [],
        'itr': [],
        'mse': [],
        'mae': []
    }

    for horizon in pred_len_list:
        print('---------------------------------------------')
        print(f'dataset {dataset}, pred_len: {horizon}, seed: {fix_seed}')
        print('---------------------------------------------')
        init_seed(fix_seed)
        mae_list, mse_list = [], []
        for ii in range(itr):
            print(f'>>>>>>>>>>>>Iteration {ii + 1}/{itr}')
            spv_config = load_spv_config(model_id, dataset=dataset)
            spv_config['data']['pred_len'] = horizon
            if horizon in [336, 720] and model_id == 'DIN_Crossformer':
                spv_config['model']['d_model'] = 16
                spv_config['model']['d_ff'] = 16

            spv = Supervisor(**spv_config)
            spv.train()
            loss = spv.test(message=f"s{fix_seed}_{ii}", plot_id=None)['loss']
            mae_list.append(loss['mae_overall'])
            mse_list.append(loss['mse_overall'])

            # update rep_rets
            rep_rets['run_id'].append(spv.get_run_id())
            rep_rets['horizon'].append(horizon)
            rep_rets['itr'].append(ii)
            rep_rets['mse'].append(loss['mse_overall'])
            rep_rets['mae'].append(loss['mae_overall'])

            pbar.update()
        print(f'>>>>>>>>>>>>MAE: {np.mean(mae_list):.4f}, MSE: {np.mean(mse_list):.4f}')
    rep_rets = pd.DataFrame(rep_rets)
    rep_rets.to_csv(f'rets/{model_id}_kdd24_{dataset}_s{fix_seed}{save_id}.csv', index=False)
    print("Train finished")


if __name__ == "__main__":
    model_id = f'DIN_{args.backbone}'
    train_kdd(model_id, itr=args.itr)
