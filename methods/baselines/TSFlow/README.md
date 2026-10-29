# TSFlow

Retrieve the pinned implementation from [TSFlow](https://github.com/marcelkollovieh/TSFlow) using the repository-root command:

```bash
python tools/fetch_baselines.py --models TSFlow
python tools/fetch_baselines.py --models TSFlow --check
```

Checkout: `third_party/upstream/TSFlow/`.

Commit: `ff893176a41a1037f63efae674aab71480eee7d0`.

Files relative to that checkout:

- `tsflow/model/tsflow_cond.py`
- `bin/train_model.py`
- `configs_local/train_conditional.yaml`

See [download behavior and model interfaces](../../../docs/UPSTREAM_BASELINES.md) and the downloaded upstream README for dependencies and data preparation.
