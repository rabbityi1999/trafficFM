"""Multivariate vector autoregression with an intercept and optional ridge."""
from pathlib import Path
import numpy as np


class VAR:
    """y[t] = intercept + sum_k y[t-k] @ coefficients[k-1].

    Every lag uses the complete sensor vector. Multi-step forecasts recurse
    on predictions, never on future observations. Ridge leaves the intercept
    unpenalized. Training sufficient statistics are accumulated in batches.
    """

    def __init__(self, lags=3, ridge=1e-3, batch_size=512):
        if isinstance(lags, bool) or not isinstance(lags, (int, np.integer)) or lags < 1:
            raise ValueError('lags must be a positive integer')
        if not np.isfinite(ridge) or ridge < 0 or batch_size < 1:
            raise ValueError('ridge must be nonnegative and batch_size positive')
        self.lags, self.ridge, self.batch_size = int(lags), float(ridge), int(batch_size)

    def fit(self, series, train_mask=None):
        series = np.asarray(series, dtype=np.float64)
        if series.ndim != 2 or series.shape[0] <= self.lags or series.shape[1] < 1:
            raise ValueError('series must have shape [time,nodes] with time > lags')
        if train_mask is None:
            train_mask = np.ones(len(series), dtype=bool)
        train_mask = np.asarray(train_mask)
        if train_mask.dtype != np.bool_ or train_mask.shape != (len(series),):
            raise ValueError('train_mask must be a boolean [time] array')
        # Every member of a regression row must be in the training set.
        ends = np.arange(self.lags, len(series))
        keep = train_mask[ends].copy()
        for lag in range(1, self.lags + 1):
            keep &= train_mask[ends-lag]
        ends = ends[keep]
        if len(ends) < 2:
            raise ValueError('at least two eligible training rows are required')
        nodes = series.shape[1]
        width = nodes*self.lags
        xx, xy = np.zeros((width, width)), np.zeros((width, nodes))
        sx, sy = np.zeros(width), np.zeros(nodes)
        for offset in range(0, len(ends), self.batch_size):
            t = ends[offset:offset+self.batch_size]
            x = np.concatenate([series[t-lag] for lag in range(1, self.lags+1)], axis=1)
            y = series[t]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError('eligible training values must be finite')
            xx += x.T@x
            xy += x.T@y
            sx += x.sum(0)
            sy += y.sum(0)
        n = len(ends)
        mean_x, mean_y = sx/n, sy/n
        gram = xx - n*np.outer(mean_x, mean_x)
        cross = xy - n*np.outer(mean_x, mean_y)
        gram = .5*(gram+gram.T)
        gram.flat[::width+1] += self.ridge
        weights = (np.linalg.solve(gram, cross) if self.ridge > 0
                   else np.linalg.lstsq(gram, cross, rcond=None)[0])
        self.coefficients_ = weights.reshape(self.lags, nodes, nodes)
        self.intercept_ = mean_y - mean_x@weights
        self.n_features_in_, self.n_training_rows_ = nodes, n
        if not np.isfinite(weights).all() or not np.isfinite(self.intercept_).all():
            raise FloatingPointError('nonfinite fitted coefficients')
        return self

    def forecast(self, history, horizon=144):
        if not hasattr(self, 'coefficients_'):
            raise RuntimeError('fit or load the model before forecasting')
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        history = np.asarray(history, dtype=np.float64)
        single = history.ndim == 2
        if single:
            history = history[None]
        if history.ndim != 3 or history.shape[1] < self.lags or history.shape[2] != self.n_features_in_:
            raise ValueError('expected [batch,history>=lags,nodes] or [history,nodes]')
        state = history[:, -self.lags:].copy()
        if not np.isfinite(state).all():
            raise ValueError('forecast history must be finite')
        predictions = []
        for _ in range(horizon):
            with np.errstate(over='raise', invalid='raise'):
                next_value = np.broadcast_to(self.intercept_, (len(state), self.n_features_in_)).copy()
                for lag in range(1, self.lags+1):
                    next_value += state[:, -lag]@self.coefficients_[lag-1]
            if not np.isfinite(next_value).all():
                raise FloatingPointError('recursive forecast is nonfinite')
            predictions.append(next_value)
            state = np.concatenate((state[:, 1:], next_value[:, None]), axis=1)
        result = np.stack(predictions, axis=1)
        return result[0] if single else result

    def save(self, path):
        if not hasattr(self, 'coefficients_'):
            raise RuntimeError('model has not been fitted')
        # A file handle prevents np.savez from silently appending a suffix.
        with Path(path).open('wb') as handle:
            np.savez(handle, version=1, lags=self.lags, ridge=self.ridge,
                     coefficients=self.coefficients_, intercept=self.intercept_,
                     n_training_rows=self.n_training_rows_)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as state:
            if int(state['version']) != 1:
                raise ValueError('unsupported checkpoint version')
            model = cls(int(state['lags']), float(state['ridge']))
            model.coefficients_ = state['coefficients'].copy()
            model.intercept_ = state['intercept'].copy()
            model.n_features_in_ = len(model.intercept_)
            model.n_training_rows_ = int(state['n_training_rows'])
        if model.coefficients_.shape != (model.lags, model.n_features_in_, model.n_features_in_):
            raise ValueError('invalid coefficient shape')
        if not np.isfinite(model.coefficients_).all() or not np.isfinite(model.intercept_).all():
            raise ValueError('nonfinite checkpoint')
        return model
