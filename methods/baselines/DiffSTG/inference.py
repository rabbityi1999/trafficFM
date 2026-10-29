# -*- coding: utf-8 -*-
import os
import torch
import argparse
import numpy as np
import torch.utils.data
from easydict import EasyDict as edict
from timeit import default_timer as timer
import contextlib
import sys

from utils.eval import Metric
from utils.common_utils import to_device, ws
from algorithm.dataset import CleanDataset, TrafficDataset
from tqdm import tqdm

def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def get_config(data_name='pems-bay'):
    config = edict()
    config.PATH_MOD = ws + '/output/model/'
    config.PATH_LOG = ws + '/output/log/'
    config.PATH_FORECAST = ws + '/output/forecast/'

    # Data Config
    config.data = edict()
    config.data.name = data_name
    baseline_root = ws
    benchmark_root = os.path.dirname(os.path.dirname(os.path.dirname(baseline_root)))
    config.data.path = os.path.join(benchmark_root, 'data', 'processed', 'DiffSTG') + '/'
    config.data.feature_file = config.data.path + config.data.name + '/flow.npy'
    config.data.spatial = config.data.path + config.data.name + '/adj.npy'
    config.data.num_recent = 1

    # Dataset specific configs
    if config.data.name == 'PEMS08':
        config.data.num_features = 1
        config.data.num_vertices = 170
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(17856 * 0.6)
        config.data.test_start_idx = int(17856 * 0.8)

    elif config.data.name == "AIR_BJ":
        config.data.num_features = 1
        config.data.num_vertices = 34
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 0.6)
        config.data.test_start_idx = int(8760 * 0.8)

    elif config.data.name == 'AIR_GZ':
        config.data.num_features = 1
        config.data.num_vertices = 41
        config.data.points_per_hour = 1
        config.data.val_start_idx = int(8760 * 10 / 12)
        config.data.test_start_idx = int(8160 * 11 / 12)

    elif config.data.name == 'metr-la':
        config.data.num_features = 3
        config.data.num_vertices = 207
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(34272 * 0.7)
        config.data.test_start_idx = int(34272 * 0.8)

    elif config.data.name == 'pems-bay':
        config.data.num_features = 3
        config.data.num_vertices = 325
        config.data.points_per_hour = 12
        config.data.val_start_idx = int(52116 * 0.7)
        config.data.test_start_idx = int(52116 * 0.8)

    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    # Model config
    config.model = edict()
    config.model.T_p = 144
    config.model.T_h = 144
    config.model.V = config.data.num_vertices
    config.model.F = config.data.num_features
    config.model.week_len = 7
    config.model.day_len = config.data.points_per_hour * 24
    config.model.d_h = 32
    config.model.N = 200
    config.model.sample_steps = 10
    config.n_samples = 8

    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.model.device = config.device

    return config

def inference(model, data_loader, config, clean_data, n_samples=8):
    setup_seed(2022)
    y_pred, y_true, time_lst = [], [], []
    metrics = Metric(T_p=config.model.T_p)
    model.eval()

    print(f"\nRunning inference with {n_samples} samples, sample_steps={config.model.sample_steps}")
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Inference", dynamic_ncols=True)

    for i, batch in pbar:
        time_start = timer()

        future, history, pos_w, pos_d = to_device(batch, config.device)
        x = torch.cat((history, future), dim=1).to(config.device)
        x_masked = torch.cat((history, torch.zeros_like(future)), dim=1).to(config.device)
        x = x.transpose(1, 3)
        x_masked = x_masked.transpose(1, 3)

        with torch.no_grad():
            if n_samples > 1:
                x_hat_list = []
                for _ in range(n_samples):
                    with suppress_stdout():
                        x_hat_single = model((x_masked, pos_w, pos_d), 1)
                    x_hat_list.append(x_hat_single)
                    torch.cuda.empty_cache()
                x_hat = torch.cat(x_hat_list, dim=1)
            else:
                with suppress_stdout():
                    x_hat = model((x_masked, pos_w, pos_d), n_samples)

        if x_hat.shape[-1] != (config.model.T_h + config.model.T_p):
            x_hat = x_hat.transpose(2, 4)

        time_lst.append((timer() - time_start))
        x, x_hat = clean_data.reverse_normalization(x), clean_data.reverse_normalization(x_hat)
        x_hat = x_hat.detach()
        f_x, f_x_hat = x[:, :, :, -config.model.T_p:], x_hat[:, :, :, :, -config.model.T_p:]

        _y_true_ = f_x.transpose(1, 3).cpu().numpy()
        _y_pred_ = f_x_hat.transpose(2, 4).cpu().numpy()

        # Evaluate only channel 0 traffic values; channel 1/2 are TOD/DOW.
        _y_true_ = _y_true_[..., :1]
        _y_pred_ = _y_pred_[..., :1]

        # Undo prepare_data.py normalization if dataset_info.txt exists.
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
            _y_true_ = _y_true_ * _prepare_std + _prepare_mean
            _y_pred_ = _y_pred_ * _prepare_std + _prepare_mean

        # Clip after converting back to raw traffic scale.
        _y_pred_ = np.clip(_y_pred_, 0, np.inf)

        metrics.update_metrics(_y_true_, _y_pred_)

        y_pred.append(_y_pred_)
        y_true.append(_y_true_)

        pbar.set_postfix({
            "MAE": f"{metrics.metrics['mae']:.4f}",
            "RMSE": f"{metrics.metrics['rmse']:.4f}",
            "Time": f"{np.sum(time_lst):.1f}s"
        })

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    time_cost = np.sum(time_lst)
    metrics.metrics['time'] = time_cost

    return metrics, y_pred, y_true

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Inference script for DiffSTG')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data', type=str, default='pems-bay', help='Dataset name')
    parser.add_argument('--sample_steps', type=int, default=10, help='Number of sampling steps')
    parser.add_argument('--n_samples', type=int, default=8, help='Number of samples for ensemble')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--output', type=str, default=None, help='Output path for predictions')
    parser.add_argument('--use_multi_gpu', type=str2bool, default=False, help='Use multiple GPUs')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs (comma separated, e.g., 0,1,2,3)')

    args = parser.parse_args()

    # Setup
    setup_seed(2022)
    torch.set_num_threads(2)

    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    # Set GPU
    if torch.cuda.is_available():
        if args.use_multi_gpu and len(gpu_ids) > 1:
            torch.cuda.set_device(gpu_ids[0])
            print(f"Using multi-GPU: {gpu_ids}")
        else:
            torch.cuda.set_device(gpu_ids[0])
            print(f"Using single GPU: {gpu_ids[0]}")

    # Load config
    config = get_config(args.data)
    config.model.sample_steps = args.sample_steps
    config.n_samples = args.n_samples
    config.use_multi_gpu = args.use_multi_gpu
    config.gpu_ids = gpu_ids
    config.device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    config.model.device = config.device

    print(f"\nConfiguration:")
    print(f"  Dataset: {config.data.name}")
    print(f"  Model path: {args.model_path}")
    print(f"  Sample steps: {config.model.sample_steps}")
    print(f"  N samples: {config.n_samples}")
    print(f"  Multi-GPU: {config.use_multi_gpu}")
    print(f"  GPU IDs: {config.gpu_ids}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {config.device}")

    # Load data
    print(f"\nLoading dataset...")
    clean_data = CleanDataset(config)
    test_dataset = TrafficDataset(clean_data, (config.data.test_start_idx + config.model.T_p, -1), config)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=False)
    print(f"  Test samples: {len(test_dataset)}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    try:
        model = torch.load(args.model_path, map_location=config.device, weights_only=False)

        # Handle DataParallel wrapped model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model = model.to(config.device)

        # Wrap with DataParallel for multi-GPU
        if config.use_multi_gpu and len(config.gpu_ids) > 1:
            print(f"  Wrapping model with DataParallel on GPUs: {config.gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            config.is_parallel = True
        else:
            config.is_parallel = False

        # Set sample steps
        model_to_use = model.module if config.is_parallel else model
        if hasattr(model_to_use, 'set_ddim_sample_steps'):
            model_to_use.set_ddim_sample_steps(args.sample_steps)

        print("  Model loaded successfully!")
    except Exception as e:
        print(f"  Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run inference
    print(f"\n{'='*60}")
    print(f"Starting Inference")
    print(f"{'='*60}")
    if config.use_multi_gpu and len(config.gpu_ids) > 1:
        print(f"Tip: Consider increasing --batch_size to {len(config.gpu_ids) * 4} or higher to fully utilize multiple GPUs\n")

    metrics, y_pred, y_true = inference(model, test_loader, config, clean_data, n_samples=args.n_samples)

    # Print results
    print(f"\n{'='*60}")
    print(f"Inference Results")
    print(f"{'='*60}")
    print(f"  MAE:  {metrics.metrics['mae']:.4f}")
    print(f"  RMSE: {metrics.metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics.metrics['mape']:.4f}")
    print(f"  Time: {metrics.metrics['time']:.2f}s")
    print(f"{'='*60}")

    # Print step-wise metrics
    print(f"\nStep-wise metrics:")
    metrics.print_specific_steps(steps=[12, 36, 72, 144])

    # Save results
    if args.output is None:
        output_path = ws + '/output/forecast/inference_results.pkl'
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    import pickle
    with open(output_path, 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'y_true': y_true,
            'metrics': metrics.to_dict(),
            'config': {
                'data': args.data,
                'sample_steps': args.sample_steps,
                'n_samples': args.n_samples
            }
        }, f)

    print(f"\nResults saved to: {output_path}")

    # Save CSV
    csv_path = output_path.replace('.pkl', '_metrics.csv')
    metrics.save_step_metrics_to_csv(csv_path)
    print(f"Metrics saved to: {csv_path}")

if __name__ == '__main__':
    main()
