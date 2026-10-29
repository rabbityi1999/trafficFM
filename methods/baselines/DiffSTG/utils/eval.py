# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from timeit import default_timer as timer


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Metric(object):
    """Computes and stores the average and current value,调用时先reset"""

    def __init__(self, T_p):
        self.time_start = timer()
        self.T_p = T_p
        self.best_metrics = {'mae': np.inf, 'rmse': np.inf, 'mape': np.inf, 'crps':np.inf, 'mis': np.inf, 'epoch': np.inf}
        self.metrics = {}
        # Store per-step metrics
        self.step_metrics = {'mae': [], 'rmse': [], 'mape': []}


    def update_metrics(self, y_true, y_pred):
        """
        both y_true and y_pred should be numpy
        :param y_true: (B, T_p, V, D)
        :param y_pred: (B, n_samples, T_p, V, D) or (B, T_p, V, D)
        :return:
        """
        assert  isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), \
            f"y_true and y_pred should be np.ndarray, now its type is y_true:{type(y_true)}, y_pred:{type(y_pred)}"
        self.metrics = {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0, 'crps': 0, 'mis': 0.0, 'time': 0.0}

        if y_pred.shape == y_true.shape: #  y_true: (B, T_p, V, D) and y_pred: (B, T_p, V, D)
            y_pred = np.expand_dims(y_pred, axis = 1) # (B, 1, T_p, V, D)

        # probabilistic metric
        eval_points = np.ones_like(y_true)
        self.metrics['crps'] = calc_quantile_CRPS(torch.from_numpy(y_true), torch.from_numpy(y_pred), torch.from_numpy(eval_points))
        self.metrics['mis'] = calc_mis(torch.from_numpy(y_true), torch.from_numpy(y_pred))

        # Calculate per-step metrics (before averaging samples)
        # y_pred: (B, n_samples, T_p, V, D), y_true: (B, T_p, V, D)
        self.step_metrics = {'mae': [], 'rmse': [], 'mape': []}
        for t in range(y_true.shape[1]):  # For each time step
            y_true_t = y_true[:, t:t+1, :, :]  # (B, 1, V, D)
            y_pred_t = y_pred[:, :, t:t+1, :, :]  # (B, n_samples, 1, V, D)
            y_pred_t_mean = np.mean(y_pred_t, axis=1)  # (B, 1, V, D)

            mae_t, rmse_t, mape_t, _ = self.get_metric(y_true_t, y_pred_t_mean)
            self.step_metrics['mae'].append(mae_t)
            self.step_metrics['rmse'].append(rmse_t)
            self.step_metrics['mape'].append(mape_t)

        # deterministic metric (overall)
        y_pred = np.mean(y_pred, axis=1) # # (B, T_p, V, D)

        self.metrics['mae'], self.metrics['rmse'], self.metrics['mape'], mse = self.get_metric(y_true, y_pred)
        # self.metrics['time'] = time_to_str((timer() - self.time_start))
        self.metrics['time'] = timer() - self.time_start



    def update_best_metrics(self, epoch=0):
        self.best_metrics['mae'], mae_state = self.get_best_metric(self.best_metrics['mae'], self.metrics['mae'])
        self.best_metrics['rmse'], rmse_state = self.get_best_metric(self.best_metrics['rmse'], self.metrics['rmse'])
        self.best_metrics['mape'], mape_state = self.get_best_metric(self.best_metrics['mape'], self.metrics['mape'])
        self.best_metrics['crps'], crps_state = self.get_best_metric(self.best_metrics['crps'], self.metrics['crps'])
        self.best_metrics['mis'], mis_state = self.get_best_metric(self.best_metrics['mis'], self.metrics['mis'])

        if mae_state:
            self.best_metrics['epoch'] = int(epoch)

    @staticmethod
    def get_metric(y_true, y_pred):
        mae = masked_mae_np(y_true, y_pred, 0)
        mse = masked_mse_np(y_true, y_pred, 0)
        mape = masked_mape_np(y_true, y_pred, 0)
        rmse = mse ** 0.5
        return mae, rmse, mape, mse

    @staticmethod
    def get_best_metric(best, candidate):
        state = False
        if candidate < best:

            best = candidate
            state = True
        return best, state

    def __str__(self):
        """For print"""
        return f"{self.metrics['mae']:<7.2f}{self.metrics['rmse']:<7.2f}{self.metrics['mape']:<7.2f}{self.metrics['crps']:<7.2f} {self.metrics['mis']:<7.2f} | {self.best_metrics['epoch'] + 1:<4} "

    def best_str(self):
        """For save"""
        return f"{self.best_metrics['epoch']},{self.best_metrics['mae']:.2f},{self.best_metrics['rmse']:.2f},{self.best_metrics['mape']:.2f}"


    def log_lst(self, epoch=None, sep=','):
        message_lst = []
        index = ['mae', 'rmse', 'mape']

        for i in index:
            message_lst.append(f"{i},{self.multi_step_str(obj=i, sep=sep, epoch=epoch)}")
        return message_lst

    def to_dict(self):
        return self.metrics

    def save_step_metrics_to_csv(self, save_path):
        """
        Save per-step metrics to CSV file
        :param save_path: path to save CSV file
        """
        df = pd.DataFrame({
            'step': list(range(1, len(self.step_metrics['mae']) + 1)),
            'mae': self.step_metrics['mae'],
            'rmse': self.step_metrics['rmse'],
            'mape': self.step_metrics['mape']
        })
        df.to_csv(save_path, index=False)
        print(f"Per-step metrics saved to {save_path}")

    def print_specific_steps(self, steps=[12, 36, 72, 144]):
        """
        Print metrics for specific steps and average
        :param steps: list of steps to print (1-indexed)
        """
        print("\n" + "="*80)
        print("Metrics at Specific Horizons")
        print("="*80)
        print(f"{'Horizon':<10}{'MAE':<10}{'RMSE':<10}{'MAPE':<10}{'CRPS':<10}{'MIS':<10}")
        print("-"*80)

        # Print specific steps
        for step in steps:
            if step <= len(self.step_metrics['mae']):
                idx = step - 1  # Convert to 0-indexed
                print(f"{step:<10}{self.step_metrics['mae'][idx]:<10.4f}"
                      f"{self.step_metrics['rmse'][idx]:<10.4f}"
                      f"{self.step_metrics['mape'][idx]:<10.4f}"
                      f"{'-':<10}{'-':<10}")

        # Print average (overall metrics)
        print(f"{'Average':<10}{self.metrics['mae']:<10.4f}"
              f"{self.metrics['rmse']:<10.4f}"
              f"{self.metrics['mape']:<10.4f}"
              f"{self.metrics['crps']:<10.4f}"
              f"{self.metrics['mis']:<10.4f}")
        print("="*80)


# metric for Probabilistic evaluation
import torch
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """

    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def MIS(
        target: np.ndarray,
        lower_quantile: np.ndarray,
        upper_quantile: np.ndarray,
        alpha: float,


) -> float:
    r"""
    mean interval score
    Implementation comes form glounts.evalution metrics
    .. math::
    msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U])
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )
    return numerator


def calc_mis(target, forecast, alpha = 0.05):
    """
       target: (B, T, V),
       forecast: (B, n_sample, T, V)
    """
    return MIS(
        target = target.cpu().numpy(),
        lower_quantile = torch.quantile(forecast, alpha / 2, dim=1).cpu().numpy(),
        upper_quantile = torch.quantile(forecast, 1.0 - alpha / 2, dim=1).cpu().numpy(),
        alpha = alpha,
    )


if __name__ == "__main__":
    # test for CRPS
    B, T, V = 32, 12, 36
    n_sample = 100
    target = torch.randn((B, T, V))
    forecast = torch.randn((B, n_sample, T, V))
    label = target.unsqueeze(1).expand_as(forecast)
    eval_points = torch.randn_like(target)

    crps = calc_quantile_CRPS(target, forecast, eval_points)
    print('crps:', crps)

    crps = calc_quantile_CRPS(target, label, eval_points)
    print('crps:', crps)

    mis = calc_mis(target, forecast)
    print('mis:', mis)

    mis = calc_mis(target, label)
    print('mis:', mis)

