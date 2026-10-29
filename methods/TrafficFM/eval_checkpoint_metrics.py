import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import util
from engine import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--adjdata", type=str, required=True)
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=144)
    parser.add_argument("--in_dim", type=int, default=3)
    parser.add_argument("--num_nodes", type=int, default=207)
    parser.add_argument("--nhid", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--output", type=str, required=True, help="output .npz containing raw-scale pred and true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_module = importlib.import_module(f"models.{args.model}")

    print("loading dataset...", flush=True)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    print("dataset loaded", flush=True)
    scaler = dataloader["scaler"]

    print("loading adjacency/model...", flush=True)
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    supports = model_module.make_supports(adj_mx, device)

    engine = trainer(
        scaler,
        args.in_dim,
        args.seq_length,
        args.num_nodes,
        args.nhid,
        args.dropout,
        0.0001,
        0.0005,
        device,
        supports,
        False,
        False,
        None,
        model_module=model_module,
        train_objective="model",
    )

    state = torch.load(args.checkpoint, map_location=device)
    engine.model.load_state_dict(state)
    engine.model.eval()
    print("checkpoint loaded", flush=True)

    realy = torch.tensor(dataloader["y_test"], dtype=torch.float32, device=device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # [samples, nodes, horizon]

    outputs = []
    print("evaluating test set...", flush=True)
    batch_no = 0
    with torch.no_grad():
        for x, y in dataloader["test_loader"].get_iterator():
            batch_no += 1
            if batch_no == 1 or batch_no % 100 == 0:
                print(f"eval batch {batch_no}", flush=True)
            testx = torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 2, 1)
            testx = F.pad(testx, (1, 0, 0, 0))
            pred = engine.model(testx)          # [B, T, N, 1]
            pred = pred.transpose(1, 3)         # [B, 1, N, T]
            outputs.append(pred.squeeze(1))     # [B, N, T]

    yhat = torch.cat(outputs, dim=0)[: realy.size(0), ...]

    # Keep the same raw scale used by the original test metrics.
    pred_raw = scaler.inverse_transform(yhat).detach().cpu().numpy().transpose(0, 2, 1).astype(np.float32)
    true_raw = realy.detach().cpu().numpy().transpose(0, 2, 1).astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, pred=pred_raw, true=true_raw)
    print(f"Saved full test predictions to {output_path} | shape={pred_raw.shape}", flush=True)

    mae, mape, rmse = [], [], []
    for i in range(args.seq_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        m = util.metric(pred, real)
        mae.append(m[0])
        mape.append(m[1])
        rmse.append(m[2])

        if i + 1 in (36, 72, 144):
            print(
                f"{i+1:>3}  MAE={m[0]:.6f}  RMSE={m[2]:.6f}  MAPE={m[1]:.6f}",
                flush=True,
            )

    print(
        f"Average  MAE={np.mean(mae):.6f}  RMSE={np.mean(rmse):.6f}  MAPE={np.mean(mape):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
