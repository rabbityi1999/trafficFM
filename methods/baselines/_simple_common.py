"""Fixed-index speed data and streaming point metrics for Linear and VAR."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def causal_fill(values, observed):
    values, observed = np.asarray(values), np.asarray(observed)
    if values.ndim != 2 or observed.shape != values.shape or observed.dtype != np.bool_:
        raise ValueError('expected values and boolean mask of shape [time,nodes]')
    result = np.empty(values.shape, dtype=np.float64)
    last = np.zeros(values.shape[1], dtype=np.float64)
    for t in range(len(values)):
        valid = observed[t] & np.isfinite(values[t])
        last = np.where(valid, values[t], last)
        result[t] = last
    return result


class TrafficData:
    def __init__(self, directory, history=144, horizon=144):
        directory = Path(directory)
        names = ('data_in144_out144.pkl', 'index_in144_out144.pkl', 'scaler_in144_out144.pkl')
        # Pickle inputs must originate from a trusted dataset preparation process.
        loaded = []
        for name in names:
            with (directory/name).open('rb') as handle:
                loaded.append(pickle.load(handle))
        packed, indices, scaler = loaded
        speed = np.asarray(packed['processed_data'], dtype=np.float64)
        if speed.ndim != 3 or speed.shape[2] < 1:
            raise ValueError('processed_data must be [time,nodes,features]')
        speed = speed[..., 0]
        self.mean, self.std = float(scaler['args']['mean']), float(scaler['args']['std'])
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError('scaler must have finite mean and positive std')
        self.raw = speed*self.std+self.mean
        self.observed = np.isfinite(self.raw) & (np.abs(self.raw)>1e-5)
        self.values = causal_fill(speed, self.observed)
        self.history, self.horizon = int(history), int(horizon)
        if history < 1 or horizon < 1:
            raise ValueError('history and horizon must be positive')
        self.indices = {}
        for split in ('train', 'val', 'test'):
            source = indices.get(split, indices.get('valid') if split == 'val' else None)
            a = np.asarray(source)
            if a.ndim != 2 or a.shape[1] != 3 or len(a) == 0 or not np.issubdtype(a.dtype, np.integer):
                raise ValueError(f'{split} must be a nonempty integer [windows,3] index')
            if ((a[:, 0]<0).any() or (a[:, 2]>len(speed)).any()
                    or not np.all(a[:, 1]-a[:, 0] == history)
                    or not np.all(a[:, 2]-a[:, 1] == horizon)):
                raise ValueError(f'invalid bounds or window lengths in {split}')
            self.indices[split] = a.astype(np.int64)
        for first, second in (('train', 'val'), ('val', 'test')):
            if self.indices[first][:, 2].max() > self.indices[second][:, 1].min():
                raise ValueError('split target ranges must be chronologically disjoint')
        self.train_mask = np.zeros(len(speed), dtype=bool)
        for start, _, end in self.indices['train']:
            self.train_mask[start:end] = True
        self.file_hashes = {name:sha256(directory/name) for name in names}

    def batches(self, split, batch_size=32, order=None):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        indices = self.indices[split]
        if order is not None:
            indices = indices[order]
        for offset in range(0, len(indices), batch_size):
            rows = indices[offset:offset+batch_size]
            x = np.stack([self.values[s:m] for s,m,e in rows])
            y = np.stack([self.raw[m:e] for s,m,e in rows])
            mask = np.stack([self.observed[m:e] for s,m,e in rows])
            yield x, y, mask


def evaluate(predict, data, split, batch_size):
    horizons = sorted(set([h for h in (36, 72, 144, data.horizon) if h <= data.horizon]))
    totals = {str(h):np.zeros(4, dtype=np.float64) for h in horizons}
    totals['Average'] = np.zeros(4, dtype=np.float64)
    for x, target, mask in data.batches(split, batch_size):
        prediction = np.asarray(predict(x))*data.std+data.mean
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError('predictions must be finite and match [batch,horizon,nodes]')
        for name, where in [(str(h),h-1) for h in horizons]+[('Average',slice(None))]:
            keep = mask[:, where]
            truth = target[:, where][keep]
            error = prediction[:, where][keep]-truth
            totals[name] += [len(truth), np.abs(error).sum(), np.square(error).sum(),
                             (np.abs(error)/np.abs(truth)).sum()]
    result = {}
    for name, (n, sae, sse, ape) in totals.items():
        if n == 0:
            raise ValueError(f'no observed targets for {split}/{name}')
        result[name] = {'MAE':float(sae/n), 'RMSE':float(np.sqrt(sse/n)),
                        'MAPE_percent':float(100*ape/n), 'n_observed':int(n)}
    return result


def save_result(path, method, args, data, metrics, checkpoint, details, source_files):
    payload = {'method':method, 'implementation_version':1, 'forecast_type':'point',
               'configuration':{k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
               'dataset_sha256':data.file_hashes, 'checkpoint_sha256':sha256(checkpoint),
               'source_sha256':{p.name:sha256(p) for p in source_files},
               'metrics_original_scale':metrics, **details}
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return payload
