"""EMA of the T2L weights, warm-up ``min(decay, (1 + n) / (10 + n))`` after n updates.

One shadow per unique Parameter, so a tensor shared by two modules stays one shadow.
"""

import torch


class ModelEMA:
    def __init__(self, named_modules, decay=0.9999):
        self.decay = decay
        self.updates = 0
        self.modules = list(named_modules)
        self.params = []  # (param, shadow)
        self._shadow_by_id = {}
        for _, m in self.modules:
            for p in m.parameters():
                if id(p) not in self._shadow_by_id:
                    s = p.detach().clone()
                    self._shadow_by_id[id(p)] = s
                    self.params.append((p, s))

    def current_decay(self):
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self):
        d = self.current_decay()
        self.updates += 1
        for p, s in self.params:
            if p.requires_grad:
                s.lerp_(p, 1.0 - d)  # s = d * s + (1 - d) * p
            else:
                s.copy_(p)

    def state_dicts(self):
        """Per-module state dicts (shadow params, current buffers) in the raw checkpoint layout."""
        out = {"decay": self.decay, "updates": self.updates}
        for key, m in self.modules:
            sd = m.state_dict()
            for name, p in m.named_parameters(remove_duplicate=False):
                sd[name] = self._shadow_by_id[id(p)]
            out[key] = {k: v.detach().clone() for k, v in sd.items()}
        return out

    @torch.no_grad()
    def load_state_dicts(self, ema_state):
        """Missing or reshaped entries keep the current weights, like the trainer's resume."""
        self.updates = int(ema_state.get("updates", 0))
        for key, m in self.modules:
            sd = ema_state.get(key, {})
            for name, p in m.named_parameters(remove_duplicate=False):
                v = sd.get(name)
                if v is not None and v.shape == p.shape:
                    self._shadow_by_id[id(p)].copy_(v)
