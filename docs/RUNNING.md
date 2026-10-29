# Data and execution

## Data layout

The unified traffic runners consume the existing fixed-index data format:

```text
data/processed/STD-MAE/METR-LA/
data/processed/STD-MAE/PEMS-BAY/
    data_in144_out144.pkl
    index_in144_out144.pkl
    scaler_in144_out144.pkl
    adj_mx.pkl                 # required by graph-dependent runners
```

`data_in144_out144.pkl` contains `processed_data` with shape `[time,node,feature]`. Feature 0 is standardized speed. TrafficFM's usual three features are speed, fractional time-of-day, and integer day-of-week. `scaler_in144_out144.pkl` contains `args.mean` and `args.std`. The index dictionary has `train`, `valid` (or `val`, where supported), and `test` arrays of `(start, history_end, future_end)` triples. Both intervals must have length 144. Preserve the exact original split files and sensor ordering; do not generate a new split and call it the same benchmark.

Only load pickle files you trust. Dataset files and sensor metadata are not supplied in this repository. Obtain METR-LA and PEMS-BAY through their authorized source, and use the corresponding benchmark preprocessing/index artifacts. Any newly generated indices require a separately identified experiment. Fit scaling, graph statistics and calibration only on permitted training/validation information.

## TrafficFM

From `methods/TrafficFM/`, inspect the CLI first:

```bash
python train_pems.py --help
python train_metr.py --help
```

Example for the independent data-space implementation (not a paper-table reproduction command):

```bash
python train_pems.py --model data_space_fixed --train_objective model \
  --unified_data ../../data/processed/STD-MAE/PEMS-BAY \
  --adjdata ../../data/processed/STD-MAE/PEMS-BAY/adj_mx.pkl \
  --devices 0 --num_nodes 325 --seq_length 144 --in_dim 3 \
  --blocks 2 --layers 2 --nhid 16 --batch_size 4 \
  --epochs 10 --seed 99 --save ../../outputs/trafficfm_data_space/ \
  --skip_final_test
```

The training entry points require CUDA. Specify the data, graph, objective, model, seed and architecture explicitly. `eval_probabilistic_144.py` is an archived legacy PEMS-BAY-specific evaluator, not a generic evaluator for repaired checkpoints. Read its fixed model/architecture assumptions before use. Do not use it to evaluate a different model variant merely because tensor shapes match.

## Unified baseline runners

To obtain DCRNN, DGCRN, GWNET, STGCN, HimNet, STD-MAE, TSFlow, UniST, STUM and FreqFlow, run `python tools/fetch_baselines.py --all` from the repository root. Their pinned upstream model paths and native training interfaces are described in [UPSTREAM_BASELINES.md](UPSTREAM_BASELINES.md).

From the repository root:

```bash
python methods/baselines/CSDI/run_unified_144.py --help
python methods/baselines/DiffSTG/run_unified_144.py --help
python methods/baselines/STID/run_unified_144.py --help
```

These runners resolve the fixed-index data directories relative to the repository. Use `--dataset METR-LA` or `--dataset PEMS-BAY` and set `--output-dir` explicitly. Budget caps and validation caps in archived defaults identify development runs; they must be recorded, not silently treated as full-budget settings.

Additional packages include `pyyaml`, `linear-attention-transformer` (CSDI), `easydict`, `einops` (DiffSTG), and the packages imported by each selected model. Install these in a separate baseline environment as needed. This release has not rerun full traffic training.

## Linear and VAR

Linear and VAR have standalone fixed-index runners using `--dataset-dir` and a new `--output-dir`. See [Linear](../methods/baselines/Linear/README.md) for Adam training with validation selection and [VAR](../methods/baselines/VAR/README.md) for closed-form fitting and recursive forecasts. Both save their configuration, checkpoint, data/source hashes and original-scale point metrics.

## Time-Series-Library and PatchSTG

Time-Series-Library includes its runner, model modules, layers, data providers and captured requirements. Some optional models require substantially different environments. Select only the model being reproduced. The captured requirements are not a tested lockfile for all models.

PatchSTG includes the model, loader, training source and the archived `METRLA144_patchstg.conf` / `PEMSBAY144_patchstg.conf` development configurations. These reference the benchmark's HimNet-format data, DiffSTG-format adjacency and graph-proxy metadata. The spatial metadata and data artifacts are not bundled; provide the matching files before running. These configurations do not establish a checkpoint-to-paper-row correspondence.

## Evaluation records

Record raw and calibrated metrics separately, including target masks and aggregation. In sample evaluations, K is the number of solver steps and M is the number of independent trajectories. Existing 64-origin development evaluations are not an untouched confirmatory test set. A point baseline does not acquire a predictive distribution solely by appearing in a probabilistic results table; record any residual or ensemble adapter explicitly.
