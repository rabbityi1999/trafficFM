# Data and execution

## Data layout

Download links for METR-LA (`metr-la.h5`), PEMS-BAY (`pems-bay.h5`) and graph metadata are provided in [DATASETS.md](DATASETS.md).

The unified traffic runners consume the existing fixed-index data format:

```text
data/processed/STD-MAE/METR-LA/
data/processed/STD-MAE/PEMS-BAY/
    data_in144_out144.pkl
    index_in144_out144.pkl
    scaler_in144_out144.pkl
    adj_mx.pkl                 # required by graph-dependent runners
```

`data_in144_out144.pkl` contains `processed_data` with shape `[time,node,feature]`. Feature 0 is standardized speed. TrafficFM's usual three features are speed, fractional time-of-day, and integer day-of-week. `scaler_in144_out144.pkl` contains `args.mean` and `args.std`. The index dictionary has `train`, `valid` (or `val`, where supported), and `test` arrays of `(start, history_end, future_end)` triples. Both intervals must have length 144. Benchmark comparisons require the original split files and sensor ordering.

Only load pickle files you trust. Dataset files and sensor metadata are not supplied in this repository. Obtain METR-LA and PEMS-BAY through their authorized source, and use the corresponding benchmark preprocessing/index artifacts. Any newly generated indices require a separately identified experiment. Fit scaling, graph statistics and calibration only on permitted training/validation information.

## TrafficFM

From `methods/TrafficFM/`, inspect the CLI first:

```bash
python train_pems.py --help
python train_metr.py --help
```

Development example: data-space implementation, 10 training epochs, final test evaluation disabled. For published-result checkpoint availability, see [experiment records](VERSIONS.md#experiment-records).

```bash
python train_pems.py --model data_space_fixed --train_objective model \
  --unified_data ../../data/processed/STD-MAE/PEMS-BAY \
  --adjdata ../../data/processed/STD-MAE/PEMS-BAY/adj_mx.pkl \
  --devices 0 --num_nodes 325 --seq_length 144 --in_dim 3 \
  --blocks 2 --layers 2 --nhid 16 --batch_size 4 \
  --epochs 10 --seed 99 --save ../../outputs/trafficfm_data_space/ \
  --skip_final_test
```

The training entry points require CUDA. Specify the data, graph, objective, model, seed and architecture explicitly. `eval_probabilistic_144.py` targets the archived PEMS-BAY model and architecture. Use it only with matching legacy checkpoints; other variants require a compatible evaluator.

## Unified baseline runners

To obtain DCRNN, DGCRN, GWNET, STGCN, HimNet, STD-MAE, TSFlow, UniST, STUM and FreqFlow, run `python tools/fetch_baselines.py --all` from the repository root. Their pinned upstream model paths and native training interfaces are described in [UPSTREAM_BASELINES.md](UPSTREAM_BASELINES.md).

From the repository root:

```bash
python methods/baselines/CSDI/run_unified_144.py --help
python methods/baselines/DiffSTG/run_unified_144.py --help
python methods/baselines/STID/run_unified_144.py --help
```

These runners resolve the fixed-index data directories relative to the repository. Use `--dataset METR-LA` or `--dataset PEMS-BAY` and set `--output-dir` explicitly. Archived defaults use development training and validation caps. Record these caps alongside the selected training budget.

Additional packages include `pyyaml`, `linear-attention-transformer` (CSDI), `easydict`, `einops` (DiffSTG), and the packages imported by each selected model. Install these in a separate baseline environment as needed. Validation coverage is listed in [VALIDATION.md](VALIDATION.md).

## Linear and VAR

Linear and VAR have standalone fixed-index runners using `--dataset-dir` and a new `--output-dir`. See [Linear](../methods/baselines/Linear/README.md) for Adam training with validation selection and [VAR](../methods/baselines/VAR/README.md) for closed-form fitting and recursive forecasts. Both save their configuration, checkpoint, data/source hashes and original-scale point metrics.

## Time-Series-Library and PatchSTG

Time-Series-Library includes its runner, model modules, layers, data providers and captured requirements. Some optional models require substantially different environments. Select only the model being reproduced. The captured requirements list dependencies; cross-model environment compatibility remains untested.

PatchSTG includes the model, loader, training source and the archived `METRLA144_patchstg.conf` / `PEMSBAY144_patchstg.conf` development configurations. These reference the benchmark's HimNet-format data, DiffSTG-format adjacency and graph-proxy metadata. The spatial metadata and data artifacts are not bundled; provide the matching files before running. Checkpoint mapping status is documented in [experiment records](VERSIONS.md#experiment-records).

## Evaluation records

Record raw and calibrated metrics separately, including target masks and aggregation. In sample evaluations, K is the number of solver steps and M is the number of independent trajectories. The existing 64-origin evaluations are development evaluations used during model development. Independent confirmatory evaluation requires a separate, untouched test protocol. For probability evaluation of point baselines, specify the residual or ensemble adapter that generates the predictive samples.
