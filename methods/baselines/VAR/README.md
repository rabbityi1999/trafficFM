# VAR

Multivariate vector autoregression of the complete sensor-speed vector:

```text
y[t] = intercept + sum(y[t-k] @ coefficients[k-1], k=1,...,p)
```

The coefficient tensor is `[lag, source_sensor, target_sensor]`. All sensors can influence each predicted sensor. The model fits one-step transitions and recursively forecasts the requested horizon, feeding predictions back as inputs. It has no spatial adjacency constraint or output clipping.

## Run

From the repository root, with NumPy installed:

```bash
python methods/baselines/VAR/run_unified_144.py --dataset-dir data/processed/STD-MAE/PEMS-BAY --output-dir outputs/var_bay_p3 --history 144 --horizon 144 --lags 3 --ridge 0.001 --batch-size 32
```

Use the METR-LA directory for that dataset. Each run requires a new output directory. Defaults are `p=3`, ridge `1e-3`, fitting batch size `512`, evaluation batch size `32`. The lag count cannot exceed the history length. Hyperparameters are fixed by CLI arguments; use validation results to choose configurations.

## Fitting

The objective is the **sum** of squared transition errors plus `ridge * ||coefficients||^2`; the intercept is unpenalized. Statistics are accumulated in float64 batches, centered, and solved as a regularized linear system. `--ridge 0` uses a least-squares solve of the centered normal equations. A positive ridge improves conditioning for correlated or rank-deficient lag features. The dense Gram matrix occupies `8 * (p * N)^2` bytes, where `N` is the sensor count.

The runner consumes the three trusted fixed-index pickle files described in [data layout](../../../docs/RUNNING.md). It uses standardized speed only and the supplied training-fitted scalar mean/std. Training timestamps are the union of the history and target intervals in the training windows; each transition and all its lagged inputs must lie within that union. Validation and test target ranges must be chronologically separate from earlier split targets.

Nonfinite speeds and raw speeds with absolute value at most `1e-5` are missing. Values are filled forward per sensor, with standardized zero before its first observation. VAR fits the resulting filled training series. Evaluation excludes missing targets and uses their original observed values. Forecasts depend on the last `p` available history steps and generated values, without future observations. Nonfinite recursive forecasts raise an error.

## Outputs and inference

`model.npz` stores coefficients, intercept, lag count, ridge and training-row count. `metrics.json` records configuration, source/data/checkpoint SHA-256 hashes, training-row count, and validation/test MAE, RMSE and percentage MAPE in original speed units. Numbered metric keys denote individual lead steps; `Average` aggregates all observed points over the horizon and all supplied windows. Outputs are point forecasts.

From this directory:

```python
from model import VAR

model = VAR.load("path/to/model.npz")
prediction = model.forecast(history_standardized, horizon=144)
```

History may be `[history, sensors]` or `[batch, history, sensors]`. The result replaces the history axis with the future horizon. Checkpoints use NumPy arrays and are loaded with `allow_pickle=False`.

Implementation version: 1. Tests: `python -m unittest discover -s tests -p test_simple_baselines.py -v` from the repository root.
