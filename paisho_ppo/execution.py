"""Measured execution options; model arithmetic and saved optimizer state stay fixed."""
import torch
from .runtime import defaults

OPTIONS = ('adam_foreach', 'clip_foreach', 'async_transfer')


class Execution:
    def __init__(self, device, enabled=True, options=None):
        self.device = torch.device(device)
        selected = defaults(self.device, torch.__version__).get('execution', {}) if options is None else options
        if set(selected)-set(OPTIONS) or any(type(v) is not bool for v in selected.values()):
            raise ValueError('invalid execution options')
        self.options = {name: bool(enabled and self.device.type == 'mps' and selected.get(name, False))
                        for name in OPTIONS}

    def move(self, batch):
        return {k: v.to(self.device, non_blocking=self.options['async_transfer']) for k, v in batch.items()}

    @torch.no_grad()
    def clip(self, parameters, maximum, error_if_nonfinite=True):
        parameters = list(parameters)
        gradients = [p.grad for p in parameters if p.grad is not None]
        eligible = gradients and all(g.device == self.device and g.dtype == torch.float32 for g in gradients)
        if not self.options['clip_foreach'] or not eligible:
            return torch.nn.utils.clip_grad_norm_(parameters, maximum, error_if_nonfinite=error_if_nonfinite)
        # Keep the same per-tensor norm, parameter order, final norm and scaling
        # equation. Grouped dispatch must pass the pinned-device byte gate.
        norms = torch._foreach_norm(gradients, 2.0)
        total = torch.linalg.vector_norm(torch.stack(norms), 2.0)
        if error_if_nonfinite and (total.isnan() | total.isinf()):
            raise RuntimeError('non-finite gradient norm')
        coefficient = torch.clamp(float(maximum)/(total+1e-6), max=1.0)
        torch._foreach_mul_(gradients, coefficient)
        return total

    def step(self, optimizer):
        if not self.options['adam_foreach']:
            return optimizer.step()
        # foreach is scheduling metadata, not a different saved Adam recipe.
        # Restore each group even after failure so checkpoints keep identical
        # param_groups as well as identical weights and numerical moments.
        previous = [g['foreach'] for g in optimizer.param_groups]
        try:
            for group in optimizer.param_groups:
                group['foreach'] = True
            return optimizer.step()
        finally:
            for group, value in zip(optimizer.param_groups, previous):
                group['foreach'] = value
