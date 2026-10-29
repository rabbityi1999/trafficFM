#!/usr/bin/env python3
"""Fixed-index, ensemble-evaluated CSDI traffic forecasting runner."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

METHOD_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(METHOD_ROOT))
from dataset_traffic import get_dataloader  # noqa: E402
from main_model import CSDI_Forecasting  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sums():
    return {key: {"n": 0, "sae": 0., "sse": 0., "ape": 0., "crps": 0., "wd": 0.}
            for key in ("36", "72", "144", "Average")}


def update(out, samples, target, mask):
    """Metrics for raw [B,S,H,N] CSDI sample paths."""
    prediction = samples.mean(dim=1)
    valid = mask.bool() & torch.isfinite(target) & (target.abs() > 1e-5)
    diff = prediction - target
    wd = (samples - target.unsqueeze(1)).abs().mean(dim=1)
    pairwise = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1, 2))
    crps = wd - 0.5 * pairwise
    for name, where in (("36", 35), ("72", 71), ("144", 143), ("Average", slice(None))):
        keep = valid[:, where]
        cur_diff, cur_y = diff[:, where], target[:, where]
        out[name]["n"] += int(keep.sum().item())
        out[name]["sae"] += float(cur_diff.abs()[keep].sum().item())
        out[name]["sse"] += float(cur_diff.square()[keep].sum().item())
        out[name]["ape"] += float((cur_diff.abs()[keep] / cur_y.abs()[keep]).sum().item())
        out[name]["wd"] += float(wd[:, where][keep].sum().item())
        out[name]["crps"] += float(crps[:, where][keep].sum().item())


@torch.no_grad()
def evaluate(model, loader, device, std, mean, samples, max_batches=0, progress=False):
    model.eval()
    out = sums()
    std, mean = std.view(1, 1, 1, -1), mean.view(1, 1, 1, -1)
    for batch_number, batch in enumerate(loader, start=1):
        if max_batches and batch_number > max_batches:
            break
        paths, target, target_mask, _, _ = model.evaluate(batch, samples)
        # CSDI native layout: [B,S,N,288]; the fixed future is last 144.
        paths = paths.permute(0, 1, 3, 2)[:, :, 144:]
        target = target.permute(0, 2, 1)[:, 144:]
        target_mask = target_mask.permute(0, 2, 1)[:, 144:]
        raw_paths = paths * std + mean
        raw_target = target * std.squeeze(1) + mean.squeeze(1)
        update(out, raw_paths, raw_target, target_mask)
        if progress and (batch_number == 1 or batch_number % 25 == 0):
            print("evaluation batch {}/{}".format(batch_number, len(loader)), flush=True)
    result = {}
    for name, item in out.items():
        if item["n"] == 0:
            raise RuntimeError("No valid fixed-test observations for {}".format(name))
        result[name] = {
            "MAE": item["sae"] / item["n"], "RMSE": (item["sse"] / item["n"]) ** .5,
            "MAPE": 100. * item["ape"] / item["n"], "CRPS": item["crps"] / item["n"],
            "WD": item["wd"] / item["n"],
        }
    return result


@torch.no_grad()
def validation_loss(model, loader, max_batches):
    model.eval()
    losses = []
    for number, batch in enumerate(loader, start=1):
        if max_batches and number > max_batches:
            break
        losses.append(float(model(batch, is_train=0).item()))
    return float(np.mean(losses))


def write_metrics(metrics, run_dir, metadata):
    with open(run_dir / "test_metrics.json", "w") as handle:
        json.dump({"metadata": metadata, "metrics": metrics}, handle, indent=2)
    with open(run_dir / "test_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Horizon", "MAE", "RMSE", "MAPE", "CRPS", "WD"])
        writer.writeheader()
        for horizon, values in metrics.items():
            writer.writerow({"Horizon": horizon, **values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("METR-LA", "PEMS-BAY"), required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--batches-per-epoch", type=int, default=128)
    parser.add_argument("--max-validation-batches", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional existing CSDI state_dict or unified-run checkpoint for evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/CSDI"))
    args = parser.parse_args()
    if args.test_samples < 2:
        raise ValueError("A genuine ensemble CRPS needs at least two samples")
    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA GPU, selected externally with CUDA_VISIBLE_DEVICES")
    set_seed(args.seed)
    device = "cuda:0"
    root = METHOD_ROOT.parents[2] / "data" / "processed" / "STD-MAE" / args.dataset
    data_path, index_path = root / "data_in144_out144.pkl", root / "index_in144_out144.pkl"
    # The data adapter reads the fixed index directly; no proportional split is derived.
    train_loader, valid_loader, test_loader, std, mean, target_dim = get_dataloader(
        str(data_path), str(index_path), device=device, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        history_length=144, pred_length=144,
    )
    with open(METHOD_ROOT / "config" / "traffic144_prelim.yaml") as handle:
        config = yaml.safe_load(handle)
    config["train"].update({"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                            "itr_per_epoch": args.batches_per_epoch})
    config["diffusion"]["num_steps"] = args.diffusion_steps
    config["model"]["num_sample_features"] = min(64, target_dim)
    model = CSDI_Forecasting(config, device, target_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    run_dir = args.output_dir / args.dataset.replace("-", "_") / "seed_{}".format(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or (run_dir / "best_model.pt")
    if args.evaluate_only:
        state = torch.load(checkpoint, map_location=device)
        # Older CSDI training saves a raw OrderedDict, whereas this unified
        # runner saves {"model": state_dict, ...}.  Both are valid weights.
        model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
        print("loaded {}".format(checkpoint), flush=True)
    else:
        best, stale = float("inf"), 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_values = []
            for batch_number, batch in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                loss = model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_values.append(float(loss.item()))
                if batch_number >= args.batches_per_epoch:
                    break
            valid_loss = validation_loss(model, valid_loader, args.max_validation_batches)
            record = {"epoch": epoch, "train_diffusion_loss": float(np.mean(loss_values)), "val_diffusion_loss": valid_loss}
            print(json.dumps(record), flush=True)
            with open(run_dir / "training.jsonl", "a") as handle:
                handle.write(json.dumps(record) + "\n")
            if valid_loss < best:
                best, stale = valid_loss, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    print("early stopping at epoch {}".format(epoch), flush=True)
                    break
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    metrics = evaluate(model, test_loader, device, std, mean, args.test_samples, progress=True)
    metadata = {
        "dataset": args.dataset, "protocol": "fixed STD-MAE 144-to-144 index", "speed_scale": "original",
        "test_samples": args.test_samples, "diffusion_steps": args.diffusion_steps,
        "CRPS": "empirical E|X-y|-0.5E|X-X'|", "WD": "empirical W1 to Dirac target, E|X-y|",
        "zero_labels": "excluded as missing traffic values",
    }
    write_metrics(metrics, run_dir, metadata)
    print(json.dumps(metrics, indent=2), flush=True)
    print("saved {}".format(run_dir / "test_metrics.csv"), flush=True)


if __name__ == "__main__":
    main()
