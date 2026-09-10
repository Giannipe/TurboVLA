"""Check the official EMA formula and continuity across a checkpoint resume."""
import copy
import unittest

import torch

from turbovla.training.pi05 import Pi05AdamW


def make_optimizer(value):
    param = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
    optimizer = Pi05AdamW([param], lr=0.1, weight_decay=0)
    optimizer._ema_param_names = {param: "weight"}
    return param, optimizer


class EMAResumeTests(unittest.TestCase):
    def test_official_formula(self):
        p, optimizer = make_optimizer(1.0)
        p.grad = torch.ones_like(p)
        optimizer.step()
        expected = torch.tensor([1.0]) * 0.999 + p.detach() * 0.001
        torch.testing.assert_close(optimizer._ema_params[p], expected)

    def test_resume_matches_uninterrupted_optimizer_and_ema(self):
        p, first = make_optimizer(1.0)
        p.grad = torch.ones_like(p)
        first.step()
        checkpoint = {"ema_decay": first.ema_decay,
                      "ema_model_state_dict": {"weight": first._ema_params[p].clone()}}
        q, resumed = make_optimizer(p.item())
        resumed.load_state_dict(copy.deepcopy(first.state_dict()))
        self.assertEqual(resumed.restore_ema(checkpoint), 1)
        for parameter, optimizer in ((p, first), (q, resumed)):
            parameter.grad = torch.tensor([0.5])
            optimizer.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        torch.testing.assert_close(first._ema_params[p], resumed._ema_params[q], rtol=0, atol=0)

    def test_missing_ema_is_not_silently_reset(self):
        _, optimizer = make_optimizer(1.0)
        with self.assertRaises(ValueError):
            optimizer.restore_ema({})

    def test_invalid_shape_is_rejected(self):
        _, optimizer = make_optimizer(1.0)
        with self.assertRaises(ValueError):
            optimizer.restore_ema({"ema_model_state_dict": {"weight": torch.zeros(2)}})


if __name__ == "__main__":
    unittest.main()
