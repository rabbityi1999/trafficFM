#!/usr/bin/env python3
"""Fixed-split probabilistic DiffSTG runner for METR-LA and PEMS-BAY.

This is deliberately independent of the original ``train.py``.  The latter
constructs its own 70/10/20 chronological splits and, in its evaluation path,
overwrites CRPS with zero.  This runner consumes the benchmark's existing
``data_in144_out144.pkl`` and ``index_in144_out144.pkl`` files verbatim.

Metrics are calculated on original-scale speed values.  Zeros (traffic-data
missing values) are excluded from metrics.  CRPS is the empirical ensemble
score E|X-y|-0.5E|X-X'|; WD is the empirical W1 distance from the ensemble to
the observed Dirac target, E|X-y|.  Hence neither metric is copied from MAE.
"""

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset

METHOD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHOD_ROOT))
from algorithm.diffstg.model import DiffSTG  # noqa: E402

HORIZONS = (36, 72, 144)


class IndexedTrafficWindows(Dataset):
    """Windows defined only by the benchmark-maintained index file."""

    def __init__(self, normalized_speed, raw_speed, index):
        self.normalized_speed = normalized_speed.astype(np.float32, copy=False)
        self.raw_speed = raw_speed.astype(np.float32, copy=False)
        self.index = np.asarray(index, dtype=np.int64)
        if self.index.ndim != 2 or self.index.shape[1] != 3:
            raise ValueError("Expected fixed (start, history_end, future_end) indices")
        if not np.all(self.index[:, 1] - self.index[:, 0] == 144):
            raise ValueError("The benchmark history must be exactly 144 steps")
        if not np.all(self.index[:, 2] - self.index[:, 1] == 144):
            raise ValueError("The benchmark prediction horizon must be exactly 144 steps")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        start, middle, end = self.index[item]
        # Missing zeros are forward-filled for DiffSTG's unconditional x0 loss.
        # The unmodified raw future is returned separately for metric masking.
        values = self.normalized_speed[start:end].copy()
        raw = self.raw_speed[start:end]
        missing = ~np.isfinite(raw) | (np.abs(raw) <= 1e-5)
        for step in range(values.shape[0]):
            if step == 0:
                valid = ~missing[step]
                fallback = values[step, valid].mean() if valid.any() else 0.0
                values[step, missing[step]] = fallback
            else:
                values[step, missing[step]] = values[step - 1, missing[step]]
        return (
            torch.from_numpy(values[:144, :, None]),
            torch.from_numpy(values[144:, :, None]),
            torch.from_numpy(raw[144:, :, None].copy()),
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_benchmark_data(repo_root, dataset):
    root = repo_root / "data" / "processed" / "STD-MAE" / dataset
    data = load_pickle(root / "data_in144_out144.pkl")["processed_data"]
    indices = load_pickle(root / "index_in144_out144.pkl")
    scaler = load_pickle(root / "scaler_in144_out144.pkl")["args"]
    # STD-MAE only standardizes speed (channel zero).  Keep the fixed scaler,
    # rather than computing a new one from another temporal range.
    normalized_speed = data[..., 0]
    mean, std = float(scaler["mean"]), float(scaler["std"])
    raw_speed = normalized_speed * std + mean
    _, _, adjacency = load_pickle(root / "adj_mx.pkl")
    expected_nodes = 207 if dataset == "METR-LA" else 325
    if normalized_speed.shape[1] != expected_nodes:
        raise ValueError("Unexpected data shape: {}".format(normalized_speed.shape))
    return normalized_speed, raw_speed, indices, np.asarray(adjacency, dtype=np.float32), mean, std


def build_config(num_nodes, adjacency, device, args):
    model = edict()
    model.N = args.diffusion_steps
    model.sample_steps = args.sampling_steps
    model.sample_strategy = args.sampling_strategy
    model.device = device
    model.beta_start = 0.0001
    model.beta_end = 0.02
    model.beta_schedule = "quad"
    model.epsilon_theta = "UGnet"
    model.T_h = model.T_p = 144
    model.V = num_nodes
    model.F = 1
    model.d_h = args.hidden_size
    model.C = args.hidden_size
    model.n_channels = args.hidden_size
    model.channel_multipliers = [1, 2]
    model.supports_len = 2
    model.n_blocks = 2
    model.A = adjacency
    return model


def make_condition(history, future=None):
    if future is None:
        future = torch.zeros_like(history)
    full = torch.cat([history, future], dim=1).transpose(1, 3).contiguous()
    masked = torch.cat([history, torch.zeros_like(future)], dim=1).transpose(1, 3).contiguous()
    # UGnet presently only consumes x_masked, but preserve the official tuple.
    batch_size = history.shape[0]
    positions = torch.zeros(batch_size, 288, 1, 1, dtype=torch.long, device=history.device)
    return full, (masked, positions, positions)


def new_sums():
    return {key: {"n": 0, "sae": 0.0, "sse": 0.0, "ape": 0.0, "crps": 0.0, "wd": 0.0}
            for key in ("36", "72", "144", "Average")}


def add_metrics(sums, samples, target):
    """Accumulate point and probabilistic metrics, all on raw speed scale.

    samples: [B, S, H, V], target: [B, H, V]
    """
    point = samples.mean(dim=1)
    valid = torch.isfinite(target) & (target.abs() > 1e-5)
    diff = point - target
    # Sorting yields exact 1-D W1 to the Dirac target only when using an
    # equally-weighted empirical ensemble; the mean absolute deviation below
    # is the equivalent expression E|X-y|.
    wd = (samples - target.unsqueeze(1)).abs().mean(dim=1)
    pairwise = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1, 2))
    crps = wd - 0.5 * pairwise
    for name, step in (("36", 35), ("72", 71), ("144", 143), ("Average", slice(None))):
        mask = valid[:, step]
        current_diff = diff[:, step]
        current_target = target[:, step]
        sums[name]["n"] += int(mask.sum().item())
        sums[name]["sae"] += float(current_diff.abs()[mask].sum().item())
        sums[name]["sse"] += float(current_diff.square()[mask].sum().item())
        sums[name]["ape"] += float((current_diff.abs()[mask] / current_target.abs()[mask]).sum().item())
        sums[name]["crps"] += float(crps[:, step][mask].sum().item())
        sums[name]["wd"] += float(wd[:, step][mask].sum().item())


@torch.no_grad()
def evaluate(model, loader, device, mean, std, num_samples, max_batches=0, progress=False):
    model.eval()
    sums = new_sums()
    for batch_index, (history, _future_normalized, raw_target) in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        history = history.to(device, non_blocking=True)
        raw_target = raw_target.to(device, non_blocking=True).squeeze(-1)
        _, condition = make_condition(history)
        # One trajectory at a time keeps 144-step sampling practical on 24 GB
        # GPUs and works for both DDIM and DDPM implementations.
        trajectories = []
        for _ in range(num_samples):
            generated = model(condition, n_samples=1)[:, 0, 0, :, 144:]
            trajectories.append(generated.transpose(1, 2))
        samples = torch.stack(trajectories, dim=1) * std + mean
        add_metrics(sums, samples, raw_target)
        if progress and (batch_index == 1 or batch_index % 25 == 0):
            print("evaluation batch {}/{}".format(batch_index, len(loader)), flush=True)
    result = {}
    for name, item in sums.items():
        if not item["n"]:
            raise RuntimeError("No valid labels were available for {}".format(name))
        result[name] = {
            "MAE": item["sae"] / item["n"],
            "RMSE": (item["sse"] / item["n"]) ** 0.5,
            "MAPE": 100.0 * item["ape"] / item["n"],
            "CRPS": item["crps"] / item["n"],
            "WD": item["wd"] / item["n"],
        }
    return result


def write_metrics(metrics, run_dir, metadata):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "metrics": metrics}
    with open(run_dir / "test_metrics.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(run_dir / "test_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "RMSE", "MAPE", "CRPS", "WD"])
        writer.writeheader()
        for horizon, values in metrics.items():
            writer.writerow({"Horizon": horizon, **values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--sampling-strategy", choices=("ddim_multi", "ddpm"), default="ddim_multi")
    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--max-validation-batches", type=int, default=10)
    parser.add_argument("--max-test-batches", type=int, default=0, help="diagnostic-only cap; 0 is the full fixed test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/DiffSTG"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DiffSTG formal runs require CUDA; use CUDA_VISIBLE_DEVICES outside this script")
    if args.test_samples < 2:
        raise ValueError("At least two samples are required for a valid empirical CRPS")
    set_seed(args.seed)
    device = torch.device("cuda:0")
    repo_root = METHOD_ROOT.parents[2]
    normalized, raw, index, adjacency, mean, std = load_benchmark_data(repo_root, args.dataset)
    validation_index = index.get("valid", index.get("val"))
    if validation_index is None:
        raise KeyError("The fixed index must provide valid/val")
    datasets = {
        "train": IndexedTrafficWindows(normalized, raw, index["train"]),
        "valid": IndexedTrafficWindows(normalized, raw, validation_index),
        "test": IndexedTrafficWindows(normalized, raw, index["test"]),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0),
        "valid": DataLoader(datasets["valid"], batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=0),
    }
    config = build_config(normalized.shape[1], adjacency, device, args)
    model = DiffSTG(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_dir = args.output_dir / args.dataset.replace("-", "_") / ("seed_{}".format(args.seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best_model.pt"
    if args.evaluate_only:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        print("loaded {} from epoch {}".format(checkpoint, state.get("epoch", "?")), flush=True)
    else:
        best_mae, bad_epochs = float("inf"), 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for history, future, _raw_target in loaders["train"]:
                history, future = history.to(device, non_blocking=True), future.to(device, non_blocking=True)
                full, condition = make_condition(history, future)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(full, condition)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.item()))
            # One sample is sufficient for early-stopping model selection.  It
            # is not used for formal CRPS/WD reporting.
            validation = evaluate(model, loaders["valid"], device, mean, std, num_samples=2,
                                  max_batches=args.max_validation_batches, progress=False)
            val_mae = validation["Average"]["MAE"]
            record = {"epoch": epoch, "train_diffusion_loss": float(np.mean(losses)), "val_original_mae": val_mae}
            print(json.dumps(record), flush=True)
            with open(run_dir / "training.jsonl", "a") as handle:
                handle.write(json.dumps(record) + "\n")
            if val_mae < best_mae:
                best_mae, bad_epochs = val_mae, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation": validation, "args": vars(args)}, checkpoint)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print("early stopping at epoch {}".format(epoch), flush=True)
                    break
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        print("best epoch {} (validation MAE {:.6f})".format(state["epoch"], best_mae), flush=True)
    metrics = evaluate(model, loaders["test"], device, mean, std, num_samples=args.test_samples,
                       max_batches=args.max_test_batches, progress=True)
    metadata = {
        "dataset": args.dataset, "protocol": "fixed STD-MAE 144-to-144 index", "speed_scale": "original",
        "test_samples": args.test_samples, "sampling_strategy": args.sampling_strategy,
        "diffusion_steps": args.diffusion_steps, "sampling_steps": args.sampling_steps,
        "CRPS": "empirical E|X-y|-0.5E|X-X'|", "WD": "empirical W1 to Dirac target, E|X-y|",
        "zero_labels": "excluded as missing traffic values",
    }
    write_metrics(metrics, run_dir, metadata)
    print(json.dumps(metrics, indent=2), flush=True)
    print("saved {}".format(run_dir / "test_metrics.csv"), flush=True)


if __name__ == "__main__":
    main()
