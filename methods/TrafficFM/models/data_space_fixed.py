"""TrafficFM data-space CFM repair; install as models/data_space_fixed.py.

Keep the historical encoder and U-Net, but embed/decode a DATA-space velocity
at every loss/solver evaluation. Legacy weights are initialization only.
"""
import torch
from torch import nn
from . import default as legacy

make_supports = legacy.make_supports


class DataSpaceFlow(legacy.FlowMatchingUNet):
    def __init__(self, old):
        nn.Module.__init__(self)
        for name in ('out_dim', 'patch_size', 'hidden_dim', 'num_patches'):
            setattr(self, name, getattr(old, name))
        for name in ('token_embed', 'token_decode', 'cond_proj', 'unet'):
            setattr(self, name, getattr(old, name))

    def velocity(self, state, projected_condition, t):
        hidden = self.unet(self._process_input(state.contiguous()),
                           projected_condition, t)
        return self._process_output(hidden)

    def loss(self, y_true, cond, drop_prob=0.1):
        if y_true.ndim != 4 or y_true.shape[1] != self.out_dim or y_true.shape[-1] != 1:
            raise ValueError('target must be [batch, horizon, node, 1]')
        valid = torch.isfinite(y_true)
        if not bool(valid.any()):
            raise ValueError('no observed target coordinates')
        # Invalid targets never enter embeddings/convolutions as NaN/Inf.
        # Zero is a fixed STANDARDIZED placeholder, not a future observation.
        target = torch.where(valid, y_true, torch.zeros_like(y_true)).detach()
        source = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        tb = t[:, None, None, None]
        state = (1 - tb) * source + tb * target
        projected = self.cond_proj(cond)
        if not 0 <= drop_prob <= 1:
            raise ValueError('drop_prob must be in [0, 1]')
        if self.training and drop_prob:
            keep = torch.rand(target.shape[0], device=target.device) >= drop_prob
            projected = projected * keep[:, None, None, None]
        estimate = self.velocity(state, projected, t)
        return (estimate - (target - source)).square()[valid].mean()

    def sample(self, cond, guidance_scale=0., fm_steps=20, solver='euler', track_grad=False):
        if not isinstance(fm_steps, int) or fm_steps < 1:
            raise ValueError('fm_steps must be a positive integer')
        if solver not in ('euler', 'heun', 'midpoint'):
            raise ValueError('unsupported solver')
        context = torch.enable_grad if track_grad else torch.no_grad
        with context():
            projected = self.cond_proj(cond)
            state = cond.new_empty(cond.shape[0], self.out_dim, cond.shape[2], 1).normal_()

            def field(z, time):
                t = z.new_full((z.shape[0],), time)
                v = self.velocity(z, projected, t)
                if guidance_scale > 0:
                    u = self.velocity(z, torch.zeros_like(projected), t)
                    v = u + guidance_scale * (v - u)
                return v

            dt = 1. / fm_steps
            for k in range(fm_steps):
                t = k * dt
                v = field(state, t)
                if solver == 'euler':
                    state = state + dt * v
                elif solver == 'heun':
                    state = state + .5 * dt * (v + field(state + dt * v, t + dt))
                else:
                    state = state + dt * field(state + .5 * dt * v, t + .5 * dt)
            # Already data-space output; do NOT decode it a second time.
            return state


class gwnet(legacy.gwnet):
    requires_target_mask = True
    training_space = 'standardized_speed_cfm_v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow = DataSpaceFlow(self.flow)
        # Prevent accidental strict loading between legacy/fixed semantics.
        self.register_buffer('_data_space_version', torch.tensor(1, dtype=torch.int64))

    def forward(self, input, y=None, guidance_scale=0., drop_prob=.1,
                fm_steps=20, solver='euler', target_mask=None):
        if y is not None:
            if target_mask is None or target_mask.shape != y.shape or target_mask.dtype != torch.bool:
                raise ValueError('training requires a boolean raw-target validity mask with target shape')
            y = torch.where(target_mask, y, torch.full_like(y, float('nan')))
        elif target_mask is not None:
            raise ValueError('inference must not consume a future-target mask')
        # Preserve encoder, graph regularization, and legacy forward behavior.
        return super().forward(input, y=y, guidance_scale=guidance_scale,
                               drop_prob=drop_prob, fm_steps=fm_steps, solver=solver)

    def load_legacy_initialization(self, state):
        """Explicit warm start, NOT a trained repaired checkpoint conversion."""
        if '_data_space_version' in state:
            raise ValueError('already versioned; use load_state_dict for fixed checkpoints')
        copied = dict(state)
        copied['_data_space_version'] = self._data_space_version.detach().clone()
        return self.load_state_dict(copied, strict=True)
