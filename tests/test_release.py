"""Small release-level checks using only synthetic tensors."""
import unittest
import sys
import torch
from test_repair import load, METHOD, HERE

sys.path.insert(0, str(METHOD))
calendar = load('repair_models.history_calendar_fixed', HERE/'history_calendar_fixed.py')
evaluator = load('trafficfm_eval', METHOD/'eval_probabilistic_144.py')
stid = load('stid_release', METHOD.parent/'baselines/STID/model.py')


class ReleaseTests(unittest.TestCase):
    def test_fractional_calendar_all_slots(self):
        x = torch.zeros(288, 3, 1, 1)
        x[:, 1, 0, 0] = torch.arange(288)/288
        slots, days = calendar.history_calendar_indices(x)
        torch.testing.assert_close(slots, torch.arange(288))
        torch.testing.assert_close(days, torch.zeros(288, dtype=torch.long))

    def test_calendar_invalid_encoding_rejected(self):
        x = torch.zeros(1, 3, 1, 1)
        x[:, 1] = 50
        with self.assertRaises(ValueError):
            calendar.history_calendar_indices(x)

    def test_sorted_crps_matches_pairwise(self):
        torch.manual_seed(21)
        x = torch.randn(2, 7, 12, 3)
        y = torch.randn(2, 12, 3)
        actual, first = evaluator.empirical_crps(x, y)
        expected = (x-y[:, None]).abs().mean(1) - .5*(x[:, :, None]-x[:, None, :]).abs().mean((1, 2))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(first, (x-y[:, None]).abs().mean(1))

    def test_crps_degenerate_distribution(self):
        y = torch.ones(1, 12, 3)
        x = torch.zeros(1, 4, 12, 3)
        actual, _ = evaluator.empirical_crps(x, y)
        torch.testing.assert_close(actual, torch.ones_like(y))

    def test_stid_shape_and_gradient(self):
        model = stid.STID(num_nodes=8, input_len=12, output_len=24)
        x = torch.randn(2, 12, 8, 3)
        x[..., 1] = 50/288
        x[..., 2] = 2
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 24, 8, 1))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(model.regression_layer.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
