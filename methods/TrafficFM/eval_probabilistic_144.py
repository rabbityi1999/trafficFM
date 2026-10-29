#!/usr/bin/env python3
"""Streaming native-sample CRPS/WD evaluation for TrafficFM."""
import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import util
from engine import trainer


def empirical_crps(samples, target):
    count = samples.shape[1]
    first = (samples - target.unsqueeze(1)).abs().mean(dim=1)
    ordered, _ = samples.sort(dim=1)
    coef = 2 * torch.arange(count, device=samples.device, dtype=samples.dtype) - count + 1
    pairwise = 2.0 * (ordered * coef.view(1, -1, 1, 1)).sum(dim=1) / float(count * count)
    return first - .5 * pairwise, first


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--adjdata', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--fm-steps', type=int, default=10)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=99)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError('CRPS requires at least two samples')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:0')
    data = util.load_unified_dataset(args.dataset_dir, args.batch_size, args.batch_size, args.batch_size)
    scaler = data['scaler']
    model_module = importlib.import_module('models.default')
    _, _, adj = util.load_adj(args.adjdata, 'doubletransition')
    supports = model_module.make_supports(adj, device)
    engine = trainer(scaler, 3, 144, 325, 16, .5, 1e-4, 5e-4, device, supports,
                     True, True, supports[0], model_module=model_module,
                     train_objective='model', blocks=args.blocks, layers=args.layers)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    engine.model.load_state_dict(state)
    model = engine.model.eval()
    sums = {name: {'n': 0, 'mae': 0., 'crps': 0., 'wd': 0.}
            for name in ('36', '72', '144', 'Average')}
    loader = data['test_loader']
    remaining = loader.size
    for number, (x, y) in enumerate(loader.get_iterator(), start=1):
        take = min(len(x), remaining)
        remaining -= take
        history = torch.as_tensor(x[:take], dtype=torch.float32, device=device).permute(0, 3, 2, 1)
        target = torch.as_tensor(y[:take, ..., 0], dtype=torch.float32, device=device)
        history = F.pad(history, (1, 0, 0, 0))
        trajectories = []
        with torch.inference_mode():
            for _ in range(args.samples):
                normalized = model(history, fm_steps=args.fm_steps).squeeze(-1)
                trajectories.append(scaler.inverse_transform(normalized))
        samples = torch.stack(trajectories, dim=1)  # [B,S,T,N]
        point = samples.mean(dim=1)
        crps, wd = empirical_crps(samples, target)
        valid = torch.isfinite(target) & (target.abs() > 1e-5)
        for name, selection in (('36', 35), ('72', 71), ('144', 143), ('Average', slice(None))):
            mask = valid[:, selection]
            sums[name]['n'] += int(mask.sum().item())
            sums[name]['mae'] += float((point[:, selection] - target[:, selection]).abs()[mask].sum().item())
            sums[name]['crps'] += float(crps[:, selection][mask].sum().item())
            sums[name]['wd'] += float(wd[:, selection][mask].sum().item())
        if number == 1 or number % 50 == 0:
            print('evaluation batch {}/{}'.format(number, loader.num_batch), flush=True)
    metrics = {name: {'MAE': item['mae']/item['n'], 'CRPS': item['crps']/item['n'], 'WD': item['wd']/item['n']}
               for name, item in sums.items()}
    payload = {'method': 'TrafficFM', 'protocol': 'fixed STD-MAE 144-to-144 test split',
               'samples': args.samples, 'fm_steps': args.fm_steps,
               'CRPS': "empirical E|X-y|-0.5E|X-X'|",
               'WD': 'empirical W1 from predictive ensemble to observed Dirac, E|X-y|',
               'checkpoint': args.checkpoint, 'metrics_original_speed_scale': metrics}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(payload, indent=2))
    with open(args.output_dir / 'metrics.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Horizon', 'MAE', 'CRPS', 'WD'])
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({'Horizon': name, **values})
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
