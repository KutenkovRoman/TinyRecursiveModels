import math

import torch
import torch.optim as optim
import torch.nn as nn

from typing import Type, Tuple, Dict, Any, Optional, Callable


class AdamWGrafting:

    def __init__(
        self, params,
        eps: float,
        lr: float,
        adam_betas: Tuple[float, float],
        weight_decay: float,
        adam_eps: float,
        optim_cls: Type[optim.Optimizer],
        optim_kwargs: Optional[Dict[str, Any]],
    ):
        self.eps = eps
        self.lr = lr
        self.adam_betas = adam_betas
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps

        self.active_optim = optim_cls(params, **optim_kwargs)
        self.step_counter = 0

        # This implementation does not support multiple parameter groups!!!
        self.param_groups = [{'lr': lr, 'params': None}]

        self._init_momentum_buffers()

    def _init_momentum_buffers(self):
        for group in self.active_optim.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    self.active_optim.state[param] = {
                        'grafting_m': torch.zeros_like(param.data),
                        'grafting_v': torch.zeros_like(param.data),
                        'grafting_buffer': torch.zeros_like(param.data),
                    }
    
    @torch.no_grad()
    def step(self, closure=None):
        self.step_counter += 1

        adam_lr = self.param_groups[0]['lr']
        beta_1, beta_2 = self.adam_betas
        adam_eps = self.adam_eps

        bias_correction_1 = 1 - beta_1 ** self.step_counter
        bias_correction_2_sqrt = math.sqrt(1 - beta_2 ** self.step_counter)

        adam_update_norm = 0.0

        for group in self.active_optim.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    continue

                state = self.active_optim.state[param]
                m = state['grafting_m']
                v = state['grafting_v']
                p = state['grafting_buffer']

                # Update momentum buffers
                m.lerp_(param.grad, 1 - beta_1)
                v.mul_(beta_2).addcmul_(param.grad, param.grad, value=(1 - beta_2))
                p.copy_(param.data)

                numer =  (adam_lr / bias_correction_1) * m
                denom = (v.sqrt() / bias_correction_2_sqrt).add_(adam_eps)

                # Compute Adam update norm
                u = torch.div(numer, denom)
                adam_update_norm += torch.norm(u, p='fro').item() ** 2

        adam_update_norm = math.sqrt(adam_update_norm)

        self.active_optim.step()  # this update must not involve weight decay

        active_update_norm = 0.0

        for group in self.active_optim.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    continue

                state = self.active_optim.state[param]
                p = state['grafting_buffer']

                # Compute update norm
                u = param.data - p
                active_update_norm += torch.norm(u, p='fro').item() ** 2

                # Revert update and store it in buffer
                param.data.sub_(u)
                p.copy_(u)

        active_update_norm = math.sqrt(active_update_norm)

        for group in self.active_optim.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    continue

                # Decoupled weight decay (here we use param before it was updated)
                param.data.mul_(1 - adam_lr * self.weight_decay)

                state = self.active_optim.state[param]
                u = state['grafting_buffer']

                param.data.add_(u, alpha=(adam_update_norm / (active_update_norm + self.eps)))
    
    def zero_grad(self, set_to_none: bool = False):
        self.active_optim.zero_grad(set_to_none)

    def state_dict(self):
        return self.active_optim.state_dict()

    def load_state_dict(self, state_dict):
        # Load the optimizer state
        outer_optim_state = state_dict['outer_optim_state']
        inner_optim_state = state_dict['inner_optim_state']

        self.outer_optim.load_state_dict(outer_optim_state)
        self.inner_optim.load_state_dict(inner_optim_state)
