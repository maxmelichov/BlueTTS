"""Exponential moving average of the T2L weights (text encoder, vector field, reference encoder).

One shadow per unique Parameter, so a tensor shared by two modules stays one shadow.
Decay warm-up: ``min(decay, (1 + n) / (10 + n))`` with n = number of updates so far.
``state_dicts()`` returns full per-module state dicts (shadow params + current buffers) in
the same key layout as the raw checkpoint entries.
"""

import torch
import torch.nn as nn


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


class ModelEMA:
    def __init__(self, named_modules, decay: float = 0.9999, warmup: bool = True):
        self.decay = float(decay)
        self.warmup = warmup
        self.updates = 0
        self.modules = [(k, _unwrap_module(m)) for k, m in named_modules]
        self.params = []  # (param, shadow)
        self._shadow_by_id = {}
        for _, m in self.modules:
            for p in m.parameters():
                if id(p) in self._shadow_by_id or not p.is_floating_point():
                    continue
                s = p.detach().clone()
                self._shadow_by_id[id(p)] = s
                self.params.append((p, s))

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self):
        d = self.current_decay()
        self.updates += 1
        for p, s in self.params:
            if p.requires_grad:
                s.lerp_(p.detach(), 1.0 - d)  # s = d * s + (1 - d) * p
            else:
                s.copy_(p.detach())

    def state_dicts(self) -> dict:
        out = {"decay": self.decay, "updates": self.updates}
        for key, m in self.modules:
            sd = {}
            name_to_param = dict(m.named_parameters(remove_duplicate=False))
            for k, v in m.state_dict().items():
                p = name_to_param.get(k)
                s = self._shadow_by_id.get(id(p)) if p is not None else None
                sd[k] = (s if s is not None else v).detach().clone()
            out[key] = sd
        return out

    @torch.no_grad()
    def load_state_dicts(self, ema_state: dict) -> int:
        """Load shadows saved by ``state_dicts``; missing or mismatched entries keep the
        current model value. Returns the number of tensors loaded."""
        self.updates = int(ema_state.get("updates", 0))
        n = 0
        for key, m in self.modules:
            sd = ema_state.get(key)
            if sd is None:
                continue
            for name, p in m.named_parameters(remove_duplicate=False):
                s = self._shadow_by_id.get(id(p))
                v = sd.get(name)
                if s is None or v is None or v.shape != s.shape:
                    continue
                s.copy_(v.to(s.device, s.dtype))
                n += 1
        return n
