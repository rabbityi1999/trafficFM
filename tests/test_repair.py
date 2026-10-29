"""CPU regression tests on actual archived TrafficFM architecture/engine."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
import torch

METHOD = Path(__file__).resolve().parents[1] / 'methods' / 'TrafficFM'
HERE = METHOD / 'models'
AUDIT = METHOD


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pkg = types.ModuleType('repair_models')
pkg.__path__ = [str(HERE)]
sys.modules[pkg.__name__] = pkg
legacy = load('repair_models.default', AUDIT / 'models/default.py')
fixed = load('repair_models.data_space_fixed', HERE / 'data_space_fixed.py')
sys.modules['model'] = types.ModuleType('model')  # unused engine wildcard import
util = load('util', AUDIT / 'util.py')
engine = load('repair_engine', METHOD / 'engine.py')


class RepairTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(6432)
        self.kw = dict(device='cpu', num_nodes=8, in_dim=3, out_dim=144,
                       gcn_bool=False, addaptadj=False, blocks=2, layers=2,
                       residual_channels=16, dilation_channels=16,
                       skip_channels=128, end_channels=256)
        self.model = fixed.gwnet(**self.kw).eval()
        self.x = torch.randn(1, 3, 8, 145)
        self.x[:, 1] = 50
        self.x[:, 2] = 2
        self.y = torch.randn(1, 144, 8, 1)
        self.mask = torch.ones_like(self.y, dtype=torch.bool)

    def test_original_bug_and_fixed_optimizer_update(self):
        old = legacy.gwnet(**self.kw).eval()
        old(self.x, y=self.y, drop_prob=0).backward()
        self.assertIsNone(old.flow.token_decode.weight.grad)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()
                  if 'token_decode' in n}
        loss = self.model(self.x, y=self.y, target_mask=self.mask, drop_prob=0)
        loss.backward()
        for n, p in self.model.named_parameters():
            if 'token_decode' in n or 'token_embed' in n:
                self.assertIsNotNone(p.grad, n)
                self.assertTrue(torch.isfinite(p.grad).all(), n)
                self.assertGreater(p.grad.norm().item(), 0, n)
        opt.step()
        for n, p in self.model.named_parameters():
            if n in before:
                self.assertFalse(torch.equal(before[n], p), n)

    def test_encoder_condition_gradient(self):
        self.model(self.x, y=self.y, target_mask=self.mask, drop_prob=0).backward()
        for name in ('flow.cond_proj.weight', 'start_conv.weight'):
            p = dict(self.model.named_parameters())[name]
            self.assertGreater(p.grad.norm().item(), 0)

    def test_missing_values_invariant_and_finite(self):
        mask = self.mask.clone()
        mask[:, 5:15] = False
        losses, grads = [], []
        for replacement in (float('nan'), float('inf'), 100000.):
            y = self.y.clone()
            y[~mask] = replacement
            self.model.zero_grad(set_to_none=True)
            torch.manual_seed(19)
            loss = self.model(self.x, y=y, target_mask=mask, drop_prob=0)
            loss.backward()
            losses.append(loss.detach())
            grads.append(self.model.flow.token_decode.weight.grad.clone())
        for i in (1, 2):
            self.assertTrue(torch.equal(losses[0], losses[i]))
            self.assertTrue(torch.equal(grads[0], grads[i]))
        self.assertTrue(torch.isfinite(losses[0]))

    def test_invalid_mask_and_empty_targets_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.x, y=self.y)
        with self.assertRaises(ValueError):
            self.model(self.x, y=self.y, target_mask=self.mask.float())
        with self.assertRaises(ValueError):
            self.model(self.x, y=self.y, target_mask=torch.zeros_like(self.mask))
        with self.assertRaises(ValueError):
            self.model(self.x, target_mask=self.mask)

    def test_state_space_axes_and_independent_samples(self):
        calls = []
        handle = self.model.flow.token_embed.register_forward_pre_hook(
            lambda m, a: calls.append(tuple(a[0].shape)))
        self.model(self.x, y=self.y, target_mask=self.mask, drop_prob=0)
        self.assertEqual(calls, [(1, 12, 8, 12)])
        calls.clear()
        a = self.model(self.x, fm_steps=3)
        self.assertEqual(calls, [(1, 12, 8, 12)] * 3)
        b = self.model(self.x, fm_steps=3)
        handle.remove()
        self.assertEqual(a.shape, self.y.shape)
        self.assertTrue(torch.isfinite(a).all())
        self.assertFalse(torch.equal(a, b))

    def test_one_step_sampler_is_data_euler(self):
        cond = self.model.forward_with_intermediates(self.x)[-1]
        flow = self.model.flow
        torch.manual_seed(23)
        z = torch.randn_like(self.y)
        expected = z + flow.velocity(z, flow.cond_proj(cond), z.new_zeros(1))
        torch.manual_seed(23)
        actual = flow.sample(cond, fm_steps=1)
        self.assertTrue(torch.equal(expected, actual))

    def test_solver_variants_and_differentiable_sample(self):
        cond = self.model.forward_with_intermediates(self.x)[-1]
        for solver in ('euler', 'heun', 'midpoint'):
            y = self.model.flow.sample(cond, fm_steps=2, solver=solver)
            self.assertTrue(torch.isfinite(y).all())
            self.assertFalse(y.requires_grad)
        y = self.model.flow.sample(cond, fm_steps=2, track_grad=True)
        y.square().mean().backward()
        self.assertGreater(self.model.flow.token_decode.weight.grad.norm().item(), 0)
        with self.assertRaises(ValueError):
            self.model(self.x, fm_steps=0)
        with self.assertRaises(ValueError):
            self.model(self.x, solver='unknown')

    def test_guidance_scale_one_matches_conditional_field(self):
        torch.manual_seed(23)
        conditional = self.model(self.x, fm_steps=2, guidance_scale=0)
        torch.manual_seed(23)
        guided = self.model(self.x, fm_steps=2, guidance_scale=1)
        self.assertTrue(torch.allclose(conditional, guided, rtol=1e-5, atol=1e-5))

    def test_checkpoint_semantics_and_roundtrip(self):
        old = legacy.gwnet(**self.kw)
        with self.assertRaises(RuntimeError):
            self.model.load_state_dict(old.state_dict())
        self.model.load_legacy_initialization(old.state_dict())
        with self.assertRaises(RuntimeError):
            old.load_state_dict(self.model.state_dict())
        other = fixed.gwnet(**self.kw).eval()
        other.load_state_dict(self.model.state_dict())
        torch.manual_seed(99)
        a = self.model(self.x, fm_steps=2)
        torch.manual_seed(99)
        b = other(self.x, fm_steps=2)
        self.assertTrue(torch.equal(a, b))

    def make_engine(self, module=fixed, objective='model'):
        return engine.trainer(util.StandardScaler(60., 10.), 3, 144, 8, 16,
                              0., 1e-4, 0., 'cpu', None, False, False, None,
                              model_module=module, train_objective=objective,
                              blocks=2, layers=2)

    def test_real_engine_mask_and_parameter_update(self):
        trainer = self.make_engine()
        before = trainer.model.flow.token_decode.weight.detach().clone()
        raw = self.y.squeeze(-1).transpose(1, 2) * 10 + 60
        raw[:, :, 0] = 0  # benchmark missing sentinel, before normalization
        captured = []
        hook = trainer.model.register_forward_pre_hook(
            lambda m, args, kwargs: captured.append(kwargs.get('target_mask')),
            with_kwargs=True)
        result = trainer.train(self.x[..., 1:], raw)
        hook.remove()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result))
        self.assertFalse(captured[0][:, 0].any())
        self.assertTrue(captured[0][:, 1:].all())
        self.assertFalse(torch.equal(before, trainer.model.flow.token_decode.weight))

    def test_wrong_engine_objective_rejected_legacy_unchanged(self):
        with self.assertRaises(ValueError):
            self.make_engine(objective='point')
        legacy_trainer = self.make_engine(module=legacy)
        self.assertEqual(legacy_trainer.train_objective, 'model')
        self.assertFalse(hasattr(legacy_trainer.model, 'requires_target_mask'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
