"""Train a direct Linear forecaster on fixed traffic windows."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _simple_common import TrafficData, evaluate, save_result
from model import Linear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--history', type=int, default=144)
    parser.add_argument('--horizon', type=int, default=144)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error('epochs, patience and batch-size must be positive')
    if not np.isfinite([args.lr, args.weight_decay]).all() or args.lr <= 0 or args.weight_decay < 0:
        parser.error('lr must be positive and weight-decay nonnegative and finite')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    data = TrafficData(args.dataset_dir, args.history, args.horizon)
    model = Linear(args.history, args.horizon, data.values.shape[1], args.individual).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir/'best.pt'
    history, best, stale = [], float('inf'), 0

    def predict(x):
        model.eval()
        with torch.no_grad():
            return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()

    for epoch in range(1, args.epochs+1):
        model.train()
        squared_error, count = 0., 0
        order = rng.permutation(len(data.indices['train']))
        for x, raw, mask in data.batches('train', args.batch_size, order):
            if not mask.any():
                continue
            target = np.where(mask, (raw-data.mean)/data.std, 0.)
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            target = torch.as_tensor(target, dtype=torch.float32, device=device)
            keep = torch.as_tensor(mask, device=device)
            optimizer.zero_grad(set_to_none=True)
            error = (model(x)-target)[keep]
            loss = error.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('training loss is nonfinite')
            loss.backward()
            optimizer.step()
            squared_error += loss.item()*error.numel()
            count += error.numel()
        if count == 0:
            raise ValueError('no observed training targets')
        val = evaluate(predict, data, 'val', args.batch_size)
        score = val['Average']['MAE']
        row = {'epoch':epoch, 'train_mse_standardized':squared_error/count, 'val_mae':score}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best, stale = score, 0
            torch.save({'state_dict':model.state_dict(), 'epoch':epoch,
                        'input_len':args.history, 'output_len':args.horizon,
                        'num_nodes':data.values.shape[1], 'individual':args.individual}, checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    selected = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(selected['state_dict'])
    metrics = {split:evaluate(predict, data, split, args.batch_size) for split in ('val','test')}
    (args.output_dir/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    save_result(args.output_dir/'metrics.json', 'Linear', args, data, metrics, checkpoint,
                {'selected_epoch':selected['epoch'], 'selection':'validation Average MAE',
                 'training_objective':'observed-target MSE in standardized speed'},
                [Path(__file__), HERE/'model.py', HERE.parent/'_simple_common.py'])
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
