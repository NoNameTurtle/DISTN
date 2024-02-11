import os
import yaml

abs_work_dir = 'D:/Code_Files/Pycharm/DISTN'  # absolute path of the project dir
base_model_dir = abs_work_dir + '/data/models'
base_data_dir = abs_work_dir + '/data/dataset'


def get_model_config_path(model_name, model_dir):
    return os.path.join(model_dir, f'{model_name}.yaml')


def load_spv_config(model_name='DISTN_Linear', dataset=None):
    work_dir = abs_work_dir

    model_dir = f'{work_dir}/data/model_configs' if dataset is None else f'{work_dir}/data/model_configs/{dataset}'
    with open(get_model_config_path(model_name, model_dir=model_dir)) as f:
        supervisor_config = yaml.safe_load(f)
    supervisor_config['log_dir'] = base_model_dir + supervisor_config['log_dir']
    supervisor_config['data']['root_path'] = base_data_dir + supervisor_config['data']['root_path']
    return supervisor_config
