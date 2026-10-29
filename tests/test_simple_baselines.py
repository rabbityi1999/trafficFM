"""Synthetic CPU checks for Linear, VAR, data boundaries and CLI execution."""
import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'methods'/'baselines'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


Linear = module('simple_linear', BASE/'Linear'/'model.py').Linear
VAR = module('simple_var', BASE/'VAR'/'model.py').VAR
common = module('simple_common', BASE/'_simple_common.py')


def dataset(folder):
    rng = np.random.default_rng(14)
    values = rng.normal(scale=.1, size=(40, 2, 1))
    indices = {'train':[(s,s+4,s+7) for s in range(14)],
               'val':[(16,20,23),(19,23,26)], 'test':[(22,26,29),(25,29,32)]}
    for name, value in [('data', {'processed_data':values}), ('index',indices),
                        ('scaler',{'args':{'mean':50., 'std':10.}})]:
        with (folder/(name+'_in144_out144.pkl')).open('wb') as handle:
            pickle.dump(value, handle)
    return values, indices


class LinearTests(unittest.TestCase):
    def test_shared_axes(self):
        model = Linear(3, 2, 2)
        with torch.no_grad():
            model.projection.weight.copy_(torch.tensor([[1.,0,0],[0,0,2.]]))
            model.projection.bias.zero_()
        x = torch.arange(12.).reshape(2,3,2)
        expected = torch.stack((x[:,0], 2*x[:,2]), dim=1)
        torch.testing.assert_close(model(x), expected)
        torch.testing.assert_close(model(x[...,None]), expected[...,None])

    def test_individual_and_gradients(self):
        model = Linear(3, 2, 2, individual=True)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
            model.weight[0,:,0] = 1.
            model.weight[1,:,2] = 2.
        x = torch.arange(1.,13.).reshape(2,3,2).requires_grad_()
        y = model(x)
        torch.testing.assert_close(y[:,:,0], x[:,0,0,None].expand(-1,2))
        torch.testing.assert_close(y[:,:,1], 2*x[:,2,1,None].expand(-1,2))
        y.square().mean().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0.)
        self.assertGreater(model.weight.grad.abs().sum().item(), 0.)

    def test_invalid_axes(self):
        model = Linear(4,3,2)
        for shape in ((2,3,4), (2,4,3), (2,4,2,2)):
            with self.assertRaises(ValueError):
                model(torch.zeros(shape))


class VARTests(unittest.TestCase):
    def test_fit_matches_design_solution(self):
        x = np.random.default_rng(8).normal(size=(80,3))
        model = VAR(2, ridge=.4, batch_size=7).fit(x)
        design = np.concatenate((x[1:-1],x[:-2]), axis=1)
        centered = design-design.mean(0)
        y = x[2:]
        w = np.linalg.solve(centered.T@centered+.4*np.eye(6), centered.T@(y-y.mean(0)))
        np.testing.assert_allclose(model.coefficients_.reshape(6,3), w, atol=1e-12)
        np.testing.assert_allclose(model.intercept_, y.mean(0)-design.mean(0)@w, atol=1e-12)

    def test_training_mask_excludes_future_and_gaps(self):
        x = np.random.default_rng(9).normal(size=(40,2))
        keep = np.zeros(40, dtype=bool)
        keep[:20] = True
        keep[8] = False
        model = VAR(2).fit(x, keep)
        altered = x.copy()
        altered[~keep] = np.nan
        other = VAR(2).fit(altered, keep)
        np.testing.assert_array_equal(model.coefficients_, other.coefficients_)
        self.assertEqual(model.n_training_rows_, 15)

    def test_recursive_direction_and_lag_order(self):
        model = VAR(2).fit(np.random.default_rng(3).normal(size=(20,2)))
        model.coefficients_ = np.array([[[0.,2.],[3.,0.]], [[.5,0.],[0.,.5]]])
        model.intercept_ = np.array([1.,-1.])
        x = np.array([[1.,2.],[3.,4.]])
        first = x[-1]@model.coefficients_[0]+x[-2]@model.coefficients_[1]+model.intercept_
        second = first@model.coefficients_[0]+x[-1]@model.coefficients_[1]+model.intercept_
        expected = np.stack((first,second))
        np.testing.assert_allclose(model.forecast(x,2), expected)
        np.testing.assert_allclose(model.forecast(x[None],2), expected[None])

    def test_singular_and_roundtrip(self):
        for ridge in (0.,1e-3):
            model = VAR(2,ridge).fit(np.ones((20,3)))
            np.testing.assert_allclose(model.forecast(np.ones((3,3)),4), 1.)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'weights.npz'
                model.save(path)
                other = VAR.load(path)
                np.testing.assert_array_equal(model.forecast(np.ones((3,3)),4),
                                              other.forecast(np.ones((3,3)),4))

    def test_invalid_inputs(self):
        for lags in (0,-1,1.5,True):
            with self.assertRaises(ValueError):
                VAR(lags)
        with self.assertRaises(RuntimeError):
            VAR().forecast(np.ones((4,2)))
        with self.assertRaises(ValueError):
            VAR().fit(np.full((10,2), np.nan))


class DataAndCLITests(unittest.TestCase):
    def test_causal_fill(self):
        x = np.array([[np.nan,1.],[4.,np.nan],[9.,3.]])
        filled = common.causal_fill(x, np.isfinite(x))
        np.testing.assert_array_equal(filled, [[0.,1.],[4.,1.],[9.,3.]])
        changed = x.copy()
        changed[-1] = 999.
        np.testing.assert_array_equal(common.causal_fill(changed,np.isfinite(changed))[:-1],filled[:-1])

    def test_metrics_and_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dataset(folder)
            data = common.TrafficData(folder,4,3)
            data.raw[:] = 10.
            data.observed[:] = True
            data.raw[26,0] = np.nan
            data.observed[26,0] = False
            metrics = common.evaluate(lambda x:np.full((len(x),3,2), (12.-50.)/10.), data,'test',2)
            self.assertAlmostEqual(metrics['Average']['MAE'],2.)
            self.assertAlmostEqual(metrics['Average']['RMSE'],2.)
            self.assertAlmostEqual(metrics['Average']['MAPE_percent'],20.)
            self.assertEqual(metrics['Average']['n_observed'],11)

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _, indices = dataset(folder)
            indices['val'] = [(10,14,17)]
            with (folder/'index_in144_out144.pkl').open('wb') as handle:
                pickle.dump(indices,handle)
            with self.assertRaises(ValueError):
                common.TrafficData(folder,4,3)

    def test_linear_and_var_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dataset(folder)
            for name in ('Linear','VAR'):
                out = folder/name
                cmd = [sys.executable,str(BASE/name/'run_unified_144.py'),
                       '--dataset-dir',str(folder),'--output-dir',str(out),
                       '--history','4','--horizon','3','--batch-size','4']
                cmd += ['--epochs','2','--patience','1'] if name == 'Linear' else ['--lags','2']
                completed = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode,0,completed.stderr)
                result = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
                self.assertEqual(result['method'],name)
                self.assertEqual(set(result['metrics_original_scale']),{'val','test'})
                self.assertEqual(len(result['checkpoint_sha256']),64)
                self.assertEqual(len(result['dataset_sha256']),3)
                self.assertTrue(np.isfinite(result['metrics_original_scale']['test']['Average']['MAE']))


if __name__ == '__main__':
    unittest.main()
