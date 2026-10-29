import argparse
import csv
from pathlib import Path

import numpy as np


def parse_input(item):
    if "=" not in item:
        raise ValueError(f"Input must be name=path, got: {item}")
    name, path = item.split("=", 1)
    return name, Path(path)


def metrics(pred, true, mask=None, eps=1e-5):
    valid = np.isfinite(pred) & np.isfinite(true) & (true > eps)

    if mask is not None:
        if mask.ndim == 2:
            valid = valid & mask[:, None, :]
        elif mask.ndim == 3:
            valid = valid & mask
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

    if valid.sum() == 0:
        return np.nan, np.nan, np.nan, 0

    diff = pred - true
    mae = np.abs(diff)[valid].mean()
    rmse = np.sqrt((diff[valid] ** 2).mean())
    mape = (np.abs(diff)[valid] / np.maximum(np.abs(true[valid]), eps)).mean()
    return float(mae), float(rmse), float(mape), int(valid.sum())


def build_transition_masks(true, window=12, drop_threshold=30.0, eps=1e-3):
    # true: [num_windows, horizon, num_nodes]
    bsz, horizon, num_nodes = true.shape

    transition_window = np.zeros((bsz, num_nodes), dtype=bool)
    transition_local = np.zeros((bsz, horizon, num_nodes), dtype=bool)

    for start_idx in range(horizon - window):
        start_speed = true[:, start_idx, :]
        future = true[:, start_idx + 1:start_idx + window + 1, :]

        future_valid = future > eps
        future_for_min = np.where(future_valid, future, np.inf)
        future_min = future_for_min.min(axis=1)

        has_future = np.isfinite(future_min)
        event = (start_speed > eps) & has_future & ((start_speed - future_min) > drop_threshold)

        transition_window |= event
        transition_local[:, start_idx:start_idx + window + 1, :] = (
            transition_local[:, start_idx:start_idx + window + 1, :] | event[:, None, :]
        )

    return transition_window, transition_local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--drop_threshold", type=float, default=30.0)
    args = parser.parse_args()

    inputs = [parse_input(x) for x in args.inputs]

    first_name, first_path = inputs[0]
    first = np.load(first_path)
    true_ref = first["true"].astype(np.float32)

    transition_window, transition_local = build_transition_masks(
        true_ref,
        window=args.window,
        drop_threshold=args.drop_threshold,
    )

    event_count = int(transition_window.sum())
    event_total = int(transition_window.size)
    event_ratio = event_count / event_total

    print(f"Transition sample-node windows: {event_count} / {event_total} = {event_ratio:.6f}")
    print(f"Rule: speed drop > {args.drop_threshold} km/h within {args.window} steps")

    rows = []

    for name, path in inputs:
        z = np.load(path)
        pred = z["pred"].astype(np.float32)
        true = z["true"].astype(np.float32)

        if pred.shape != true_ref.shape or true.shape != true_ref.shape:
            raise ValueError(f"{name} shape mismatch: pred={pred.shape}, true={true.shape}, ref={true_ref.shape}")

        g_mae, g_rmse, g_mape, g_n = metrics(pred, true)
        tw_mae, tw_rmse, tw_mape, tw_n = metrics(pred, true, transition_window)
        tl_mae, tl_rmse, tl_mape, tl_n = metrics(pred, true, transition_local)

        rows.append({
            "method": name,
            "transition_sample_nodes": event_count,
            "transition_sample_node_ratio": event_ratio,
            "global_mae": g_mae,
            "global_rmse": g_rmse,
            "global_mape": g_mape,
            "transition_window_mae": tw_mae,
            "transition_window_rmse": tw_rmse,
            "transition_window_mape": tw_mape,
            "transition_window_mae_ratio": tw_mae / g_mae,
            "transition_local_mae": tl_mae,
            "transition_local_rmse": tl_rmse,
            "transition_local_mape": tl_mape,
            "transition_local_mae_ratio": tl_mae / g_mae,
            "global_values": g_n,
            "transition_window_values": tw_n,
            "transition_local_values": tl_n,
        })

        print(
            f"{name:20s} "
            f"Global MAE={g_mae:.4f} | "
            f"Transition-window MAE={tw_mae:.4f} ratio={tw_mae / g_mae:.4f} | "
            f"Transition-local MAE={tl_mae:.4f} ratio={tl_mae / g_mae:.4f}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
