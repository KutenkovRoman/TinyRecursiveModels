import torch
import torch.optim as optim
import torch.nn as nn

from typing import Type, Tuple, Dict, Any, Optional, Callable


class SNOO(optim.Optimizer):
    def __init__(
        self, params,
        learning_rate: float,
        outer_momentum: float,
        inner_optim_cls: Type[optim.Optimizer],
        inner_optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.defaults = {'eta': learning_rate, 'mu': outer_momentum}
        self.state = {}

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        if inner_optim_kwargs is None:
            inner_optim_kwargs = {}

        # Create inner optimizer
        self.inner_optim = inner_optim_cls(params, **inner_optim_kwargs)

        self.step_counter = 0
        self.accum_counter = 0

        self._init_slow_weights()

    def _init_slow_weights(self):
        """Initialize slow weights in parameter states"""
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    self.state[param] = {
                        'slow_weight': param.data.clone().detach(),
                        'momentum_buffer': torch.zeros_like(param.data),
                    }

    @property
    def param_groups(self):
        return self.inner_optim.param_groups

    @torch.no_grad()
    def step(self, sync_weights, closure=None):
        if closure is not None:
            raise NotImplementedError("I have no idea how to handle closures, implement it yourself.")

        # Step inner optimizer
        loss = self.inner_optim.step()  # should probably incorporate `closure` here

        #self.step_counter += 1
        self.accum_counter += 1

        if sync_weights:  #sync_weights and self.accum_counter > 1
            eta = self.defaults['eta']
            mu = self.defaults['mu']

            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        state = self.state[param]
                        slow_weight = state['slow_weight']
                        momentum_buffer = state['momentum_buffer']

                        # Compute pseudo-gradient, update momentum buffer and slow weight
                        pseudo_grad = slow_weight - param.data
                        momentum_buffer.mul_(mu).add_(pseudo_grad)
                        slow_weight.add_(mu * momentum_buffer + pseudo_grad, alpha=-eta)

                        # Sync fast weight to slow weight
                        param.data.copy_(slow_weight)

            #if self.step_counter > 5000 and self.accum_counter > 1:
            #    print(f"Accumulated gradient for {self.accum_counter} steps until train/step {self.step_counter}")

            self.accum_counter = 0

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of both optimizers"""
        self.inner_optim.zero_grad(set_to_none)

    def state_dict(self):
        """Return the state of the optimizer"""
        state_dict = super().state_dict()
        state_dict['inner_optim_state'] = self.inner_optim.state_dict()
        #state_dict['step_counter'] = self.step_counter
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the optimizer state"""
        inner_optim_state = state_dict.pop('inner_optim_state', None)
        #self.step_counter = state_dict.pop('step_counter', 0)

        super().load_state_dict(state_dict)

        if inner_optim_state is not None:
            self.inner_optim.load_state_dict(inner_optim_state)


class LookaheadOptimizer:
    def __init__(
        self, params,
        outer_optim_cls: Type[optim.Optimizer],
        outer_optim_kwargs: Optional[Dict[str, Any]],
        inner_optim_cls: Type[optim.Optimizer],
        inner_optim_kwargs: Optional[Dict[str, Any]],
    ):
        assert outer_optim_kwargs, "At least provide learning rate for outer optimizer"
        assert inner_optim_kwargs, "At least provide learning rate for inner optimizer"

        params_list = list(params)

        self.slow_params = []
        for p in params_list:
            if p.requires_grad:
                slow_p = p.detach().clone().requires_grad_(True)
                self.slow_params.append(slow_p)
            else:
                self.slow_params.append(None)

        self.outer_optim = outer_optim_cls([p for p in self.slow_params if p is not None], **outer_optim_kwargs)
        self.inner_optim = inner_optim_cls(params_list, **inner_optim_kwargs)

    @property
    def param_groups(self):
        return self.inner_optim.param_groups

    @torch.no_grad()
    def step(self):
        # Step inner optimizer without updating slow weights
        return self.inner_optim.step()

    @torch.no_grad()
    def sync_lookahead(self):
        # Step outer optimizer and synchronize fast weights
        for group in self.inner_optim.param_groups:
            for idx, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue

                # Get corresponding slow parameter
                slow_p = self.slow_params[idx]

                # Compute pseudo-gradient
                pseudo_grad = (slow_p.data - p.data) ##/ group['lr']

                # Assign as gradient
                if slow_p.grad is None:
                    slow_p.grad = pseudo_grad.clone()
                else:
                    slow_p.grad.copy_(pseudo_grad)

        self.outer_optim.step()

        for group in self.inner_optim.param_groups:
            for idx, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue

                slow_p = self.slow_params[idx]

                # Syncronize fast and slow weights
                p.data.copy_(slow_p.data)

        # Just to be safe
        #self.outer_optim.zero_grad()


    def zero_grad(self, set_to_none: bool = False):
        # Zero the gradients of both optimizers
        self.inner_optim.zero_grad(set_to_none)

    def state_dict(self):
        # Return the state of the optimizer
        state_dict = {
            'outer_optim_state': self.outer_optim.state_dict(),
            'inner_optim_state': self.inner_optim.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        # Load the optimizer state
        outer_optim_state = state_dict['outer_optim_state']
        inner_optim_state = state_dict['inner_optim_state']

        self.outer_optim.load_state_dict(outer_optim_state)
        self.inner_optim.load_state_dict(inner_optim_state)


class AdamOuterOptimizer(optim.Optimizer):
    def __init__(
        self, params,
        learning_rate: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
        inner_optim_cls: Type[optim.Optimizer], 
        inner_optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.defaults = {'eta': learning_rate, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        self.state = {}

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        if inner_optim_kwargs is None:
            inner_optim_kwargs = {}

        # Create inner optimizer
        self.inner_optim = inner_optim_cls(params, **inner_optim_kwargs)

        self.step_counter = 0

        self._init_slow_weights()

    def _init_slow_weights(self):
        """Initialize slow weights in parameter states"""
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    self.state[param] = {
                        'slow_weight': param.data.clone().detach(),
                        'first_moment': torch.zeros_like(param.data),
                        'second_moment': torch.zeros_like(param.data),
                    }

    @property
    def param_groups(self):
        return self.inner_optim.param_groups

    @torch.no_grad()
    def step(self, sync_weights=False, closure=None):
        if closure is not None:
            raise NotImplementedError("I have no idea how to handle closures, implement it yourself.")

        # Step inner optimizer
        loss = self.inner_optim.step()

        if sync_weights:
            self.step_counter += 1

            eta = self.defaults['eta']
            beta_1, beta_2 = self.defaults['betas']
            eps = self.defaults['eps']
            weight_decay = self.defaults['weight_decay']

            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        state = self.state[param]
                        slow_weight = state['slow_weight']
                        m = state['first_moment']
                        v = state['second_moment']

                        slow_weight.mul_(1 - eta * weight_decay)

                        # Compute pseudo-gradient, update momentum buffer and slow weight
                        pseudo_grad = slow_weight - param.data
                        m.lerp_(pseudo_grad, 1 - beta_1)
                        v.mul_(beta_2).addcmul_(pseudo_grad, pseudo_grad, value=(1 - beta_2))

                        bias_correction_1 = 1 - beta_1 ** self.step_counter
                        bias_correction_2_sqrt = (1 - beta_2 ** self.step_counter) ** 0.5
                        step_size = eta / bias_correction_1

                        denom = (v.sqrt() / bias_correction_2_sqrt).add_(eps)

                        slow_weight.addcdiv_(m, denom, value=-step_size)

                        # Sync fast weight to slow weight
                        param.data.copy_(slow_weight)

        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Zero the gradients of both optimizers"""
        self.inner_optim.zero_grad(set_to_none)

    def state_dict(self):
        """Return the state of the optimizer"""
        state_dict = super().state_dict()
        state_dict['inner_optim_state'] = self.inner_optim.state_dict()
        #state_dict['step_counter'] = self.step_counter
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the optimizer state"""
        inner_optim_state = state_dict.pop('inner_optim_state', None)
        #self.step_counter = state_dict.pop('step_counter', 0)

        super().load_state_dict(state_dict)

        if inner_optim_state is not None:
            self.inner_optim.load_state_dict(inner_optim_state)

