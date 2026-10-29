"""Fit multivariate VAR and evaluate fixed traffic windows."""
import argparse
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _simple_common import TrafficData, evaluate, save_result
from model import VAR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--history', type=int, default=144)
    parser.add_argument('--horizon', type=int, default=144)
    parser.add_argument('--lags', type=int, default=3)
    parser.add_argument('--ridge', type=float, default=1e-3)
    parser.add_argument('--fit-batch-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.lags > args.history or args.batch_size < 1:
        parser.error('lags must not exceed history; batch-size must be positive')
    model = VAR(args.lags, args.ridge, args.fit_batch_size)
    data = TrafficData(args.dataset_dir, args.history, args.horizon)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.fit(data.values, data.train_mask)
    checkpoint = args.output_dir/'model.npz'
    model.save(checkpoint)
    metrics = {split:evaluate(lambda x:model.forecast(x, args.horizon), data, split, args.batch_size)
               for split in ('val','test')}
    save_result(args.output_dir/'metrics.json', 'VAR', args, data, metrics, checkpoint,
                {'training_rows':model.n_training_rows_, 'selection':'fixed CLI hyperparameters',
                 'training_objective':'sum of squared errors plus ridge on lag coefficients',
                 'fit_values':'causally filled standardized speed'},
                [Path(__file__), HERE/'model.py', HERE.parent/'_simple_common.py'])
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
