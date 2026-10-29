# Datasets

TrafficFM uses METR-LA and PEMS-BAY traffic-speed data with five-minute sampling. The benchmark window is 144 historical steps followed by 144 prediction steps.

## Download links

| Dataset | Sensors | Raw file | Download |
|---|---:|---|---|
| METR-LA | 207 | `metr-la.h5` | [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) |
| PEMS-BAY | 325 | `pems-bay.h5` | [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) |

Both files are distributed through the shared folder linked by the [DCRNN data-preparation instructions](https://github.com/liyaguang/DCRNN#data-preparation). That page also lists a [Baidu mirror](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g). Source: the upstream README, checked on 2026-10-29. Baidu mirror availability is unverified.

Sensor ordering, location and road-graph resources are available in the upstream [sensor_graph directory](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph). Use metadata matching the selected dataset and retain its sensor ordering.

## Local layout

Store the downloaded HDF5 files locally, for example:

```text
data/raw/metr-la.h5
data/raw/pems-bay.h5
```

The unified TrafficFM runners consume the processed layout below rather than the raw HDF5 files:

```text
data/processed/STD-MAE/METR-LA/
data/processed/STD-MAE/PEMS-BAY/
    data_in144_out144.pkl
    index_in144_out144.pkl
    scaler_in144_out144.pkl
    adj_mx.pkl                 # for graph-dependent runners
```

See [data format and execution](RUNNING.md#data-layout) for the contents of these files. Use the corresponding experiment's fixed indices and training-fitted scaler. Raw downloads alone do not contain TrafficFM's processed split/index artifacts.

After retrieving STD-MAE with `python tools/fetch_baselines.py --models STD-MAE`, its upstream preprocessing source is available under `third_party/upstream/STD-MAE/scripts/data_preparation/METR-LA/` and `PEMS-BAY/`. Set the intended 144-step input/output lengths and chronological split explicitly when preparing a new experiment. Each upstream baseline's own runner may use a different data layout; its downloaded README documents that interface.

Dataset files remain outside version control. Follow the data provider's terms and cite the dataset source and relevant original methods when reporting results.
