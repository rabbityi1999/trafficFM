# Fixed-version upstream baselines

Ten baseline implementations are retrieved from seven upstream repositories. DCRNN, DGCRN, GWNET and STGCN share the STG4Traffic collection. The other six models each have a dedicated upstream repository. Full commit and tree identifiers, model paths and pin provenance are recorded in [baselines.lock.json](../third_party/baselines.lock.json).

## Download

Requirements: Python 3.9 or later, Git with sparse-checkout/partial-fetch support (tested with Git 2.39.1), and HTTPS access to GitHub. From the TrafficFM repository root:

```bash
python tools/fetch_baselines.py --list
python tools/fetch_baselines.py --all
python tools/fetch_baselines.py --all --check
```

To retrieve selected models:

```bash
python tools/fetch_baselines.py --models DCRNN DGCRN TSFlow
python tools/fetch_baselines.py --models DCRNN DGCRN TSFlow --check
```

The first two selections above share one checkout. Model names are case-sensitive. Downloads go to `third_party/upstream/<source-id>/`; `--root <directory>` changes that location. Run output is one JSON record per source with the resolved commit, tree, model names, source-file count and local path. Any failed source yields a nonzero exit status while other selected sources are still processed.

## Model and runner locations

Paths below are relative to each downloaded repository. The scripts retain their upstream interfaces and data layouts. They are upstream training entry points; TrafficFM's fixed-index adapters are separate integrations.

| Model | Source folder | Model source | Training entry / configuration |
|---|---|---|---|
| DCRNN | STG4Traffic | `TrafficSpeed/model/DCRNN/dcrnn.py` | `TrafficSpeed/DCRNN/DCRNN_Main.py`; `METRLA_DCRNN.conf`, `PEMSBAY_DCRNN.conf` in the same directory |
| DGCRN | STG4Traffic | `TrafficSpeed/model/DGCRN/net.py` | `TrafficSpeed/DGCRN/DGCRN_Main.py`; dataset `.conf` files in the same directory |
| GWNET | STG4Traffic | `TrafficSpeed/model/GWNET/gwnet.py` | `TrafficSpeed/GWNET/GWNET_Main.py`; dataset `.conf` files in the same directory |
| STGCN | STG4Traffic | `TrafficSpeed/model/STGCN/stgcn.py` | `TrafficSpeed/STGCN/STGCN_Main.py`; dataset `.conf` files in the same directory |
| HimNet | HimNet | `model/HimNet.py` | `scripts/train.py`, `model/HimNet.yaml` |
| STD-MAE | STD-MAE | `stdmae/stdmae_arch/stdmae.py` | `stdmae/run.py`; `stdmae/STDMAE_METR-LA.py`, `stdmae/STDMAE_PEMS-BAY.py` |
| TSFlow | TSFlow | `tsflow/model/tsflow_cond.py` | `bin/train_model.py`, `configs_local/train_conditional.yaml` |
| UniST | UniST | `src/model.py` | `src/main.py`, `src/scripts/` |
| STUM | STUM | `src/stum/STUM.py` | `train_stum_ori.py` (vanilla), `main.py` (STGNN-enhanced) |
| FreqFlow | FreqFlow | `models/Flow_Spect.py` | `run_longExp_F.py`, `exp/exp_main_F.py` |

For STG4Traffic, use each `TrafficSpeed/<model>/` directory as the working directory and set dataset paths, mode, device, window and horizon in its configuration/CLI. For HimNet, the upstream command runs from `scripts/` and selects datasets with `-d METRLA` or `-d PEMSBAY`. STD-MAE includes S-MAE/T-MAE pretraining configurations and a downstream predictor; supply the pretraining checkpoints referenced by the chosen configuration. TSFlow uses the YAML configuration supplied to its training entry point. UniST's scripts separate pretraining and prompt tuning. STUM distinguishes its vanilla and STGNN-enhanced variants. FreqFlow uses its own time-series data loader and model-selection arguments.

Use each downloaded README and dependency files to prepare an isolated environment. The downloader does not install dependencies, prepare datasets, execute training scripts or allocate GPUs. For a 144-to-144 comparison, align preprocessing, index splits, prediction length, scaling, target masks and selection criteria in the experiment configuration.

## Version and checkout behavior

- Every fetch requests the exact 40-character commit; no branch-head update is used. `--check` makes no network request and verifies commit, tree, origin, required entry points and the Git blob hash of every selected file.
- A sparse checkout materializes source, configuration, documentation, dependency and license files. Dataset arrays, pretrained weights, images and compiled caches are excluded. The underlying partial Git checkout preserves the upstream commit identity.
- An existing matching checkout is reused. A different revision, edited tracked file, missing source file or occupied non-repository directory produces an error without resetting or overwriting it. Untracked experiment outputs are left alone.
- Fetches are staged in a uniquely named directory. Failed stages are retained at the printed path for inspection; successful stages are renamed to the source folder. Downloads are ignored by this repository's `.gitignore`.
- These dependencies remain under their upstream terms. Source retrieval does not assign them the license of TrafficFM. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Validation

All seven pinned repositories were fetched locally and their ten model mappings checked on 2026-10-29. The verification covers 613 selected files. Offline regression tests cover the ten-model inventory, shared-repository deduplication, path/revision validation, sparse source selection, content checks and protection of local edits. This is source-retrieval validation; training dependencies and datasets are configured separately per model.
