#!/usr/bin/env python3
"""Run the complete TurboVLA trainer with pi0.5 optimizer knobs."""

import argparse
import os
import sys

import torch.optim as torch_optim

from . import trainer


EMA_DECAY = 0.999
_ACTIVE_OPTIMIZER = None
_DINOV3_PRECISION = "bf16_autocast"
_ORIGINAL_PARSE_ARGS = trainer.parse_args
_ORIGINAL_BUILD_MODEL_ARCHITECTURE = trainer.build_model_architecture
_ORIGINAL_BUILD_OPTIMIZER = trainer.build_param_group_optimizer
_ORIGINAL_TORCH_SAVE = trainer.torch.save


class Pi05AdamW(torch_optim.AdamW):
    def __init__(self, params, *args, **kwargs):
        kwargs.setdefault("betas", (0.9, 0.95))
        kwargs.setdefault("eps", 1e-8)
        super().__init__(params, *args, **kwargs)
        self.ema_decay = EMA_DECAY
        self._ema_params = {}
        self._ema_param_names = {}
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.is_floating_point():
                    self._ema_params[param] = param.detach().clone()

    def step(self, closure=None):
        result = super().step(closure=closure)
        decay = self.ema_decay
        with trainer.torch.no_grad():
            for param, ema_param in self._ema_params.items():
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        return result

    def restore_ema(self, checkpoint):
        """Restore the separately saved EMA, not fresh clones of resumed raw weights."""
        state = checkpoint.get("ema_model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("resume_mode=all requires ema_model_state_dict for the EMA optimizer")
        decay = float(checkpoint.get("ema_decay", self.ema_decay))
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"invalid EMA decay in checkpoint: {decay}")
        if len(self._ema_param_names) != len(self._ema_params):
            raise ValueError("EMA parameter names must be registered before restoring")
        for param, name in self._ema_param_names.items():
            value = state.get(name)
            if not isinstance(value, trainer.torch.Tensor) or value.shape != param.shape:
                raise ValueError(f"missing or incompatible EMA parameter: {name}")
        with trainer.torch.no_grad():
            for param, name in self._ema_param_names.items():
                self._ema_params[param].copy_(state[name])
        self.ema_decay = decay
        return len(self._ema_params)


def parse_args_with_dinov3_precision():
    global _DINOV3_PRECISION
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dinov3_precision",
        type=str,
        default="bf16_autocast",
        choices=["fp32", "bf16", "bf16_autocast"],
        help=(
            "DINOv3 compute precision. bf16_autocast keeps weights and optimizer "
            "state fp32 while autocasting only the DINOv3 forward pass."
        ),
    )
    known, remaining = parser.parse_known_args()
    _DINOV3_PRECISION = known.dinov3_precision

    original_argv = sys.argv
    sys.argv = [original_argv[0], *remaining]
    try:
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv
    args.dinov3_precision = _DINOV3_PRECISION
    return args


def patch_dinov3_precision(model, dinov3_precision):
    model.vision_encoder.set_compute_precision(dinov3_precision)


def build_model_architecture_with_dinov3_precision(args):
    model = _ORIGINAL_BUILD_MODEL_ARCHITECTURE(args)
    patch_dinov3_precision(model, args.dinov3_precision)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"dinov3_precision={args.dinov3_precision}")
    return model


def build_param_group_optimizer_with_pi05_ema(model, args):
    global _ACTIVE_OPTIMIZER
    optimizer, optimizer_summary = _ORIGINAL_BUILD_OPTIMIZER(model, args)
    param_names = {}
    for name, param in model.named_parameters():
        clean_name = name[7:] if name.startswith("module.") else name
        if param in optimizer._ema_params:
            param_names[param] = clean_name
    optimizer._ema_param_names = param_names
    _ACTIVE_OPTIMIZER = optimizer
    return optimizer, optimizer_summary


def torch_save_with_pi05_ema(obj, *args, **kwargs):
    optimizer = _ACTIVE_OPTIMIZER
    if isinstance(obj, dict) and optimizer is not None and "model_state_dict" in obj:
        model_state = obj["model_state_dict"]
        if isinstance(model_state, dict):
            ema_state = dict(model_state)
            for param, name in optimizer._ema_param_names.items():
                if name in ema_state:
                    ema_tensor = optimizer._ema_params[param].detach()
                    ema_state[name] = ema_tensor.to(device="cpu", dtype=ema_state[name].dtype)
            obj = dict(obj)
            obj["ema_decay"] = optimizer.ema_decay
            obj["ema_model_state_dict"] = ema_state
    return _ORIGINAL_TORCH_SAVE(obj, *args, **kwargs)


trainer.parse_args = parse_args_with_dinov3_precision
trainer.build_model_architecture = build_model_architecture_with_dinov3_precision
trainer.AdamW = Pi05AdamW
trainer.build_param_group_optimizer = build_param_group_optimizer_with_pi05_ema
trainer.torch.save = torch_save_with_pi05_ema


def main():
    trainer.train_model()


if __name__ == "__main__":
    main()
