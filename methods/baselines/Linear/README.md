# Linear

Direct multi-horizon linear forecasting of sensor speed. For each sensor, the model applies `y = W x + b` along the time axis. The default shares `W` and `b` across sensors; `--individual` fits a separate projection per sensor. There is no trend decomposition, graph mixing or recursive decoding.

## Run

From the repository root, with NumPy and PyTorch installed:

```bash
python methods/baselines/Linear/run_unified_144.py --dataset-dir data/processed/STD-MAE/PEMS-BAY --output-dir outputs/linear_bay_s99 --history 144 --horizon 144 --epochs 50 --patience 10 --batch-size 32 --lr 0.001 --seed 99 --device cpu
```

Use the METR-LA directory for that dataset. GPU execution is available with `--device cuda:0`. Each run requires a new output directory.

The model accepts `[batch, history, sensors]` or `[batch, history, sensors, 1]` and returns the same layout with the future horizon replacing history. Only the speed feature is consumed. Model defaults are 144 history steps, 144 future steps and shared temporal weights.

## Training and evaluation

- Input files: `data_in144_out144.pkl`, `index_in144_out144.pkl`, `scaler_in144_out144.pkl`, as described in [data layout](../../../docs/RUNNING.md). These pickle files must come from a trusted source. Scaling statistics must be fitted on training data.
- Missing speeds are nonfinite values or absolute raw speeds at most `1e-5`. History is filled forward per sensor; before the first observation the fallback is standardized zero (the training mean).
- Adam minimizes MSE on observed future speeds in standardized units. Missing targets are excluded from the loss. Defaults: learning rate `1e-3`, weight decay `0`, batch size `32`, seed `99`.
- The lowest validation Average MAE selects the checkpoint. Patience is measured in epochs without improvement. Test evaluation uses the selected checkpoint.
- MAE, RMSE and percentage MAPE use observed targets in original speed units. Numbered metric keys identify individual lead steps (36, 72, 144 when available); `Average` aggregates all observed points across the entire horizon and every supplied split window. Overlapping windows contribute separately.

Outputs: `best.pt` (weights and architecture), `history.json` (epoch losses and validation scores), and `metrics.json` (configuration, selected epoch, validation/test metrics, source/data/checkpoint SHA-256 hashes). These are point forecasts.

## Load a checkpoint

From this directory:

```python
import torch
from model import Linear

state = torch.load("path/to/best.pt", map_location="cpu", weights_only=True)
model = Linear(state["input_len"], state["output_len"],
               state["num_nodes"], state["individual"])
model.load_state_dict(state["state_dict"])
model.eval()
with torch.no_grad():
    prediction = model(history_standardized)
```

Implementation version: 1. Tests: `python -m unittest discover -s tests -p test_simple_baselines.py -v` from the repository root.
