# -*- coding: utf-8 -*-
import os, sys
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer
import time

from utils.eval import Metric
from utils.gpu_dispatch import GPU
from utils.common_utils import dir_check, to_device, ws, unfold_dict, dict_merge, GpuId2CudaId, Logger

from algorithm.dataset import CleanDataset, TrafficDataset
from algorithm.diffstg.model import DiffSTG, save2file

from tqdm import tqdm

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


# for tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir = os.path.join(os.environ["NNI_OUTPUT_DIR"], 'tensorboard'))
except:
    pass

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # model
    parser.add_argument("--epsilon_theta", type=str, default='UGnet')
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--beta_schedule", type=str, default='quad')  # uniform, quad
    parser.add_argument("--beta_end", type=float, default=0.1)
    parser.add_argument("--sample_steps", type=int, default=200)  # sample_steps
    parser.add_argument("--ss", type=str, default='ddpm') #help='sample strategy', ddpm, multi_diffusion, one_diffusion
    parser.add_argument("--T_h", type=int, default=144)

    # eval
    parser.add_argument('--n_samples', type=int, default=8)

    # train
    parser.add_argument("--is_train", type=str2bool, default=True, help="train or evaluate")
    parser.add_argument("--data", type=str, default='PEMS08')
    parser.add_argument("--mask_ratio", type=float, default=0.0) # mask of history data
    parser.add_argument("--is_test", type=str2bool, default=False)
    parser.add_argument("--nni", type=str2bool, default=False)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for testing (use smaller value to save memory)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    # multi-gpu
    parser.add_argument("--use_multi_gpu", type=str2bool, default=False)
    parser.add_argument("--gpu_ids", type=str, default='0,1', help='comma separated gpu ids, e.g., 0,1')

    args, _ = parser.parse_known_args()
    return args

def default_config(data='AIR_BJ'):
    config = edict()

    # Get baseline root directory (DiffSTG location)
    baseline_root = ws
    # Get benchmark root directory (3 levels up: DiffSTG -> baselines -> methods -> project root)
    trafficfm_root = os.path.dirname(os.path.dirname(os.path.dirname(baseline_root)))

    config.PATH_MOD = baseline_root + '/output/model/'
    config.PATH_LOG = baseline_root + '/output/log/'
    config.PATH_FORECAST = baseline_root + '/output/forecast/'

    # Data Config - point to data/processed/DiffSTG/
    config.data = edict()
    config.data.name = data
    config.data.path = os.path.join(trafficfm_root, 'data', 'processed', 'DiffSTG') + '/'

    config.data.feature_file = config.data.path + config.data.name + '/flow.npy'
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1

    if config.data.name == 'PEMS08':
        config.data.num_features = 1
        config.data.num_vertices = 170
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(17856 * 0.6)
        config.data.test_start_idx = int(17856 * 0.8)

    if config.data.name == "AIR_BJ":
        config.data.num_features = 1
        config.data.num_vertices = 34
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 0.6)
        config.data.test_start_idx = int(8760 * 0.8)

    if config.data.name == 'AIR_GZ':
        config.data.num_features = 1
        config.data.num_vertices = 41
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 10 / 12) #
        config.data.test_start_idx = int(8160 * 11 / 12)

    if config.data.name == 'metr-la':
        config.data.num_features = 3  # flow, tod, dow
        config.data.num_vertices = 207
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(34272 * 0.7)  # 23990
        config.data.test_start_idx = int(34272 * 0.8)  # 27417

    if config.data.name == 'pems-bay':
        config.data.num_features = 3  # flow, tod, dow
        config.data.num_vertices = 325
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(52116 * 0.7)  # 36481
        config.data.test_start_idx = int(52116 * 0.8)  # 41692

    # GPU configuration
    config.use_multi_gpu = False  # will be set by params
    config.gpu_ids = []

    # Single GPU mode (original behavior)
    gpu_id = GPU().get_usefuel_gpu(max_memory=3000, condidate_gpu_id=[0,1,2,3,4,6,7,8])
    #Problem in GPU() definition in gpu_dispatch.py
    print(gpu_id)
    config.gpu_id = gpu_id
    if gpu_id != None:
        cuda_id = GpuId2CudaId(gpu_id)
        torch.cuda.set_device(f"cuda:{cuda_id}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model config
    config.model = edict()

    config.model.T_p = 144
    config.model.T_h = 144
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7
    config.model.day_len = config.data.points_per_hour * 24
    config.model.device = device
    config.model.d_h = 32 #Embedding dimension D (see equation 17 in paper)

    # config for diffusion model
    config.model.N = 200 #related to 4.2 adn 4.3, but details are not clear
    config.model.sample_steps = 200
    
    config.model.epsilon_theta = 'UGnet'
    config.model.is_label_condition = True
    config.model.beta_end = 0.02
    config.model.beta_schedule = 'quad'
    config.model.sample_strategy = 'ddpm'

    config.n_samples = 1  # Reduce to 1 to save memory during test

    # config for UGnet
    config.model.channel_multipliers = [1, 2]  # The list of channel numbers at each resolution.
    config.model.supports_len = 2

    # training config
    config.model_name = 'DiffSTG'
    config.is_test = False  # Whether run the code in the test mode
    config.epoch = 300  # Number of max training epoch
    config.optimizer = "adam"
    config.lr = 1e-4  #??
    config.batch_size = 32   #??
    config.wd = 1e-5  #??
    config.early_stop = 10  #??
    config.start_epoch = 0  #??
    config.device = device
    config.logger = Logger()


    if not os.path.exists(config.PATH_MOD):
        os.makedirs(config.PATH_MOD)
    if not os.path.exists(config.PATH_LOG):
        os.makedirs(config.PATH_LOG)
    if not os.path.exists(config.PATH_FORECAST):
        os.makedirs(config.PATH_FORECAST)
    return config

def evals(model, data_loader, epoch, metric, config, clean_data, mode='Test'):
    setup_seed(2022)

    y_pred, y_true, time_lst = [], [], []
    mae_sum, mse_sum, mape_sum = 0.0, 0.0, 0.0
    metric_count, mape_count = 0, 0

    # Report paper-style horizons: 36, 72, 144, and Average.
    horizon_steps = [36, 72, 144]
    horizon_metric_sums = {
        h: {'mae': 0.0, 'mse': 0.0, 'mape': 0.0, 'count': 0, 'mape_count': 0}
        for h in horizon_steps
    }
    metrics_future = Metric(T_p=config.model.T_p)
    metrics_history = Metric(T_p=config.model.T_h)
    model.eval()

    # Handle DataParallel wrapper when accessing model attributes
    model_to_use = model.module if config.is_parallel else model
    print(model_to_use.sample_strategy)
    samples, targets = [], []
    sample_saved = 0
    sample_save_limit = 50
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        evalT1 = time.perf_counter()
        evalCT = evalT1
        evalPrevT = evalT1
        evalPrevT = evalCT

        if i > 0 and config.is_test: break
        time_start = timer()

        future, history, pos_w, pos_d = to_device(batch, config.device) # target:(B,T,V,1), history:(B,T,V,1), pos_w: (B,1), pos_d:(B,T,1)

        x = torch.cat((history, future), dim=1).to(config.device)  # in cpu (B, T, V, F), T =  T_h + T_p
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)  # (B, T, V, F)
        target_for_save = x.cpu()
        x = x.transpose(1, 3)  # (B, F, V, T)
        x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)
        
        evalCT = time.perf_counter()
        evalPrevT = evalCT

        n_samples = 1 if mode == 'Val' else config.n_samples
        # n_samples = config.n_samples

        # Use torch.no_grad() and clear cache to save memory during evaluation
        with torch.no_grad():
            # For test mode with large memory requirements, process in smaller chunks
            if mode == 'test' and n_samples > 1:
                # Process each sample separately to save memory
                x_hat_list = []
                for _ in range(n_samples):
                    x_hat_single = model((x_masked, pos_w, pos_d), 1)  # (B, 1, F, V, T)
                    x_hat_list.append(x_hat_single)
                    torch.cuda.empty_cache()
                x_hat = torch.cat(x_hat_list, dim=1)  # (B, n_samples, F, V, T)
            else:
                x_hat = model((x_masked, pos_w, pos_d), n_samples) # (B, n_samples, F, V, T)

        if sample_saved < sample_save_limit:
            keep = min(sample_save_limit - sample_saved, target_for_save.shape[0])
            targets.append(target_for_save[:keep])
            samples.append(x_hat.transpose(2, 4)[:keep].cpu())
            sample_saved += keep

        # Clear GPU cache after inference
        torch.cuda.empty_cache()

        evalCT = time.perf_counter()
        evalPrevT = evalCT
        
        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p): x_hat = x_hat.transpose(2,4)
        # assert x.shape == x_hat.shape, f"shape of x ({x.shape}) does not equal to shape of x_hat ({x_hat.shape})"

        time_lst.append((timer() - time_start))
        x, x_hat= clean_data.reverse_normalization(x), clean_data.reverse_normalization(x_hat)
        x_hat = x_hat.detach()
        f_x, f_x_hat = x[:,:,:,-config.model.T_p:], x_hat[:,:,:,:,-config.model.T_p:] # future

        evalCT = time.perf_counter()
        evalPrevT = evalCT

        _y_true_ = f_x.transpose(1, 3).cpu().numpy()  # y_true: (B, T_p, V, D)
        _y_pred_ = f_x_hat.transpose(2, 4).cpu().numpy() # y_pred: (B, n_samples, T_p, V, D)

        # channel 0 = traffic value; channel 1/2 are TOD/DOW.
        _y_true_eval = _y_true_[..., :1]
        _y_pred_eval = _y_pred_[..., :1]

        # Undo prepare_data.py normalization for raw-scale metrics.
        _os = __import__('os')
        _feature_file = getattr(clean_data, 'feature_file', '')
        _info_path = _os.path.join(_os.path.dirname(_feature_file), 'dataset_info.txt')
        _prepare_mean, _prepare_std = None, None

        if _os.path.exists(_info_path):
            with open(_info_path, 'r') as _f:
                for _line in _f:
                    if _line.startswith('mean:'):
                        _prepare_mean = float(_line.split(':', 1)[1].strip())
                    elif _line.startswith('std:'):
                        _prepare_std = float(_line.split(':', 1)[1].strip())

        if _prepare_mean is not None and _prepare_std is not None:
            _y_true_eval = _y_true_eval * _prepare_std + _prepare_mean
            _y_pred_eval = _y_pred_eval * _prepare_std + _prepare_mean

        _y_pred_eval = np.clip(_y_pred_eval, 0, np.inf)
        pred_mean = _y_pred_eval.mean(axis=1)
        err = pred_mean - _y_true_eval
        mae_sum += np.abs(err).sum()
        mse_sum += np.square(err).sum()
        metric_count += err.size
        mape_mask = np.abs(_y_true_eval) > 1e-5
        if mape_mask.any():
            mape_sum += np.abs(err[mape_mask] / _y_true_eval[mape_mask]).sum()
            mape_count += mape_mask.sum()

        for _h in horizon_steps:
            _idx = _h - 1
            if pred_mean.shape[1] > _idx:
                _pred_h = pred_mean[:, _idx:_idx + 1]
                _true_h = _y_true_eval[:, _idx:_idx + 1]
                _err_h = _pred_h - _true_h

                horizon_metric_sums[_h]['mae'] += np.abs(_err_h).sum()
                horizon_metric_sums[_h]['mse'] += np.square(_err_h).sum()
                horizon_metric_sums[_h]['count'] += _err_h.size

                _mape_mask_h = np.abs(_true_h) > 1e-5
                if _mape_mask_h.any():
                    horizon_metric_sums[_h]['mape'] += np.abs(_err_h[_mape_mask_h] / _true_h[_mape_mask_h]).sum()
                    horizon_metric_sums[_h]['mape_count'] += _mape_mask_h.sum()

        evalCT = time.perf_counter()
        evalPrevT = evalCT

        # Avoid storing the full test set in memory.

        h_x, h_x_hat = x[:, :, :, :config.model.T_h], x_hat[:, :, :, :,  :config.model.T_h]
        _y_true_ = h_x.transpose(1, 3).cpu().numpy()  # y_true: (B, T_p, V, D)
        _y_pred_ = h_x_hat.transpose(2, 4).cpu().numpy()
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)
        # Skip history metrics during full test to avoid expensive per-batch metrics.

        evalCT = time.perf_counter()
        evalPrevT = evalCT

    time_cost = np.sum(time_lst)
    metric.metrics['mae'] = mae_sum / max(metric_count, 1)
    metric.metrics['mse'] = mse_sum / max(metric_count, 1)
    metric.metrics['rmse'] = np.sqrt(metric.metrics['mse'])
    metric.metrics['mape'] = 100 * mape_sum / max(mape_count, 1)

    print('Horizon   MAE       RMSE      MAPE')
    for _h in horizon_steps:
        _s = horizon_metric_sums[_h]
        _mae = _s['mae'] / max(_s['count'], 1)
        _rmse = (_s['mse'] / max(_s['count'], 1)) ** 0.5
        _mape = 100 * _s['mape'] / max(_s['mape_count'], 1)
        print(f'{_h:<9d} {_mae:<9.4f} {_rmse:<9.4f} {_mape:<9.4f}')

    _avg_mae = metric.metrics['mae']
    _avg_rmse = metric.metrics['mse'] ** 0.5
    _avg_mape = metric.metrics['mape']
    print(f'Average   {_avg_mae:<9.4f} {_avg_rmse:<9.4f} {_avg_mape:<9.4f}')
    metric.metrics['crps'] = 0.0
    metric.metrics['mis'] = 0.0
    metric.update_best_metrics(epoch=epoch)
    metric.metrics['time'] = time_cost

    if mode == 'test': # save the prediction result to file
        samples = torch.cat(samples, dim=0)
        targets = torch.cat(targets, dim=0)
        observed_flag = torch.ones_like(targets) #(B, T, V, F)
        evaluate_flag = observed_flag
        evaluate_flag[:, -config.model.T_p:, :, :] = 1
        import pickle
        with open (config.forecast_path, 'wb') as f:
            pickle.dump([samples, targets, observed_flag, evaluate_flag], f)

        message = f"predict_path = '{config.forecast_path}'"
        config.logger.message_buffer += f"{message}\n"
        config.logger.write_message_buffer()

        # Save per-step metrics to CSV
        csv_path = config.forecast_path.replace('.pkl', '_step_metrics.csv')
        metric.save_step_metrics_to_csv(csv_path)

        # Print metrics at specific horizons
        metric.print_specific_steps(steps=[12, 36, 72, 144])


    if config.nni: nni.report_intermediate_result(metric.metrics['mae'])

    # log of performance in future prediction
    if metric.best_metrics['epoch'] == epoch:
        message = f" |[{metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}]"
    else:
        message = f" | {metric.metrics['mae']:<7.2f}{metric.metrics['rmse']:<7.2f}"
    print(message, end='', flush=False)
    config.logger.message_buffer += message

    # log of performance in historical prediction
    hist_mae = metrics_history.metrics.get("mae", 0.0)
    hist_rmse = metrics_history.metrics.get("rmse", 0.0)
    message = f" | {hist_mae:<7.2f}{hist_rmse:<7.2f}{time_cost:<5.2f}s"
    print(message, end='\n', flush=False)
    config.logger.message_buffer += f"{message}\n"

    # write log message buffer
    config.logger.write_message_buffer()

    torch.cuda.empty_cache()
    return metric


from pprint import  pprint
def main(params: dict):
    # torch.manual_seed(2022)
    setup_seed(2022)
    torch.set_num_threads(2)
    config = default_config(params['data'])

    config.is_test = params['is_test']
    config.nni = params['nni']
    config.lr = params['lr']
    config.batch_size = params['batch_size']
    config.mask_ratio = params['mask_ratio']
    config.epoch = params['epochs']  # Apply epochs from command line

    # multi-gpu configuration
    config.use_multi_gpu = params.get('use_multi_gpu', False)
    if config.use_multi_gpu:
        gpu_ids_str = params.get('gpu_ids', '0,1')
        config.gpu_ids = [int(x) for x in gpu_ids_str.split(',')]
        print(f"Using multi-GPU training on GPUs: {config.gpu_ids}")
        # Set default device to first GPU
        torch.cuda.set_device(config.gpu_ids[0])
        config.device = torch.device(f'cuda:{config.gpu_ids[0]}')
    else:
        print(f"Using single GPU: {config.device}")

    # model
    config.model.N = params['N']
    config.T_h = config.model.T_h = params['T_h']
    config.T_p = config.model.T_p =  params['T_h']  #T_h and T_p doesnt have to be the same, but here both are 12
    config.model.epsilon_theta =  params['epsilon_theta']
    config.model.sample_steps = params['sample_steps']
    config.model.d_h = params['hidden_size']
    config.model.C = params['hidden_size']
    config.model.n_channels = params['hidden_size']
    config.model.beta_end = params['beta_end']
    config.model.beta_schedule = params["beta_schedule"]
    config.model.sample_strategy = params["ss"]
    config.n_samples = params['n_samples']

    if config.model.sample_steps > config.model.N:
        print('sample steps large than N, exit')
        # nni.report_intermediate_result(50)
        nni.report_final_result(50)
        return 0


    config.trial_name = '+'.join([f"{v}" for k, v in params.items()])
    config.log_path = f"{config.PATH_LOG}/{config.trial_name}.log"

    pprint(config)
    dir_check(config.log_path)
    config.logger.open(config.log_path, mode="w")
    #log parameters
    config.logger.write(config.__str__()+'\n', is_terminal=False)

    #  data pre-processing
    # print('\n1. data pre-processing ...')
    clean_data = CleanDataset(config)
    config.model.A = clean_data.adj

    print("DiffSTG config here")
    model = DiffSTG(config.model)
    model = model.to(config.device) # add gpu

    # Multi-GPU support
    if config.use_multi_gpu and len(config.gpu_ids) > 1:
        print(f"Wrapping model with DataParallel on GPUs: {config.gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        config.is_parallel = True
    else:
        config.is_parallel = False

    # Load training dataset
    train_dataset = TrafficDataset(clean_data, (0 + config.model.T_p, config.data.val_start_idx - config.model.T_p + 1), config)
    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=True, pin_memory=True)

    val_dataset = TrafficDataset(clean_data, (config.data.val_start_idx + config.model.T_p, config.data.test_start_idx - config.model.T_p + 1), config)
    # val_dataset   = TrafficDataset(clean_data, (config.data.val_start_idx + config.model.T_p, config.data.val_start_idx + config.model.T_p + 512), config)
    # Use smaller batch size for validation with long sequences
    val_batch_size = 8 if config.model.T_h >= 144 else 64
    val_loader = torch.utils.data.DataLoader(val_dataset, val_batch_size, shuffle=False)

    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    # Use smaller batch size for testing to save memory, especially for long sequences
    test_batch_size = params.get('test_batch_size', 1) if config.model.T_h >= 144 else 64
    test_loader = torch.utils.data.DataLoader(test_dataset, test_batch_size, shuffle=False)


    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # metrics in val, and test dataset, note that we cannot evaluate the performance in the train dataset
    metrics_val = Metric(T_p=config.model.T_h + config.model.T_p)

    # Handle DataParallel wrapper when accessing model methods
    model_to_use = model.module if config.is_parallel else model
    model_path = config.PATH_MOD + config.trial_name + model_to_use.model_file_name()
    config.model_path = model_path
    config.logger.write(f"model path:{model_path}\n", is_terminal=False)
    print('model_path:', model_path)
    dir_check(model_path)

    config.forecast_path = forecast_path = config.PATH_FORECAST + config.trial_name + '.pkl'
    config.logger.write(f"forecast_path:{forecast_path}\n", is_terminal=False)
    print('forecast_path:', forecast_path)
    dir_check(forecast_path)


    # log model architecture
    print(model)
    config.logger.write(model.__str__())

    # log training process
    config.logger.write(f'Num_of_parameters:{sum([p.numel() for p in model.parameters()])}\n', is_terminal=True)
    message = "      |---Train--- |---Val Future-- -|-----Val History----|\n"
    config.logger.write(message, is_terminal=True)

    message = "Epoch | Loss  Time | MAE     RMSE    |  MAE    RMSE   Time|\n" #f"{'Type':^5}{'Epoch':^5} | {'MAE':^7}{'RMSE':^7}{'MAPE':^7}
    config.logger.write(message, is_terminal=True)


    # Train by default. Use --is_train False only when explicitly evaluating a checkpoint.
    train_start_t = timer()
    # Train and sample the data
    for epoch in range(config.epoch):
        if not params['is_train']: break
        if epoch > 1 and config.is_test: break

        n, avg_loss, time_lst = 0, 0, []
        # train diffusion model
        print("Train Epoch: ", epoch)
        tqdm_forLoop = tqdm(enumerate(train_loader))
        for i, batch in tqdm_forLoop:
            if i > 3 and config.is_test:break
            time_start =  timer()
            future, history, pos_w, pos_d = batch # future:(B, T_p, V, F), history: (B, T_h, V, F)

            # get x0
            x = torch.cat((history, future), dim=1).to(config.device) #  (B, T, V, F)

            # get x0_masked
            mask =  torch.randint_like(history, low=0, high=100) < int(config.mask_ratio * 100)# mask the history in a ratio with mask_ratio
            history[mask] = 0
            x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device) # (B, T, V, F)

            # reshape
            x = x.transpose(1,3) # (B, F, V, T)
            x_masked = x_masked.transpose(1,3) # (B, F, V, T)

            # loss calculate
            # Handle DataParallel wrapper when calling model methods
            model_to_use = model.module if config.is_parallel else model
            loss = 10 * model_to_use.loss(x, (x_masked, pos_w, pos_d))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            # calculate the moving average training loss
            n += 1
            avg_loss = avg_loss * (n - 1) / n + loss.item() / n

            time_lst.append((timer() - time_start))
            message = f"{i / len(train_loader) + epoch:6.1f}| {avg_loss:0.3f} {np.sum(time_lst):.1f}s"
            #print('\r' + message, end='', flush=True)
            tqdm_forLoop.set_description(message)

        config.logger.message_buffer += message

        try:
            writer.add_scalar('train/loss', avg_loss, epoch)
        except:
            pass
        
        print("Before Eval")
        if epoch == config.epoch - 1:
            evals(model, val_loader, epoch, metrics_val, config, clean_data, mode='Val')
            scheduler.step(metrics_val.metrics['mae'])
        print("AfterEval")
        if metrics_val.best_metrics['epoch'] == epoch:
            print('[save model]>> ', model_path)
            if config.is_parallel:
                torch.save(model.module, model_path)
            else:
                torch.save(model, model_path)
        # early stop 已禁用，因为仅最后一个 epoch 验证，中间无参考指标
        # if epoch - metrics_val.best_metrics['epoch'] > config.early_stop: break


    try:
        model = torch.load(model_path, map_location=config.device, weights_only=False)
        # If we're using multi-GPU, wrap the loaded model with DataParallel
        if config.use_multi_gpu and len(config.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
        print('best model loaded from: <<', model_path)
    except Exception as err:
        print(err)
        print('load best model failed')

    # conduct multiple-samples, then report the best
    metric_lst = []
    # Reduce sample_steps to save memory for long sequences (T_h=144, T_p=144)
    # For 144-step sequences, use fewer sampling steps
    test_sample_steps = 20 if config.model.T_h >= 144 else 40
    for sample_strategy, sample_steps in [('ddim_multi', test_sample_steps)]:
        if sample_steps > config.model.N: break

        config.model.sample_strategy = sample_strategy
        config.model.sample_steps = sample_steps

        # Handle DataParallel wrapper when calling model methods
        model_to_use = model.module if config.is_parallel else model
        model_to_use.set_ddim_sample_steps(sample_steps)
        model_to_use.set_sample_strategy(sample_strategy)

        metrics_test = Metric(T_p=config.model.T_h + config.model.T_p)
        evals(model, test_loader, epoch, metrics_test, config, clean_data, mode='test')
        message = f'sample_strategy:{sample_strategy}, sample_steps:{sample_steps} Final results in test:{metrics_test}\n'
        config.logger.write(message, is_terminal=True)

        params = unfold_dict(config)
        params = dict_merge([params, metrics_test.to_dict()])
        params['best_epoch'] = metrics_val.best_metrics['epoch']
        params['model'] = config.model.epsilon_theta
        save2file(params)
        metric_lst.append(metrics_test.metrics['mae'])

    # rename log file
    log_file, log_name = os.path.split(config.log_path)
    new_log_path = os.path.join(log_file, f"[{config.data.name}]mae{min(metric_lst):7.2f}+{log_name}")
    import shutil
    # os.rename(config.log_path, new_log_path)
    shutil.copy(config.log_path, new_log_path)
    config.log_path = new_log_path

    try:
        writer.close()
    except:
        pass

    if 'nni' in globals():
        nni.report_final_result(min(metric_lst))


# data.name	model	model.N	model.epsilon_theta	model.d_h	model.T_h	model.T_p	model.sample_strategy
# PEMS08	UGnet	200	    UGnet	            32	        12	        12	        ddpm

if __name__ == '__main__':
    import logging
    logger = logging.getLogger('training')
    print('GPU:', torch.cuda.current_device())
    params = vars(get_params())
    main(params)

