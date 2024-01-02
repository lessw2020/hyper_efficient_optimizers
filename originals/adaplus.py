# Copyright (c) Lei Guan
# Licensed under the MIT license.
# from: https://raw.githubusercontent.com/guanleics/AdaPlus/main/AdaPlus.py
"""
forked from https://github.com/pytorch/pytorch/blob/master/torch/optim/adamw.py
"""

import math
import torch
from torch.optim.optimizer import Optimizer

class AdaPlus(Optimizer):
    r"""Implements AdaPlus algorithm.
    `AdaPlus: Integrating Nesterov Momentum and Precise Stepsize Adjustment on AdamW Basis`.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdaPlus, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaPlus, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)#, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_bar'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)#, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)#, memory_format=torch.preserve_format)

                exp_avg, exp_avg_bar, exp_avg_sq = state['exp_avg'], state['exp_avg_bar'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # TODO add by lei
                exp_avg_bar = beta1 * exp_avg + (1-beta1) * grad
                grad_residual = grad - exp_avg

                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq.add_(group['eps']), out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    #denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    denom = max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                else:
                    # denom = (exp_avg_sq.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    denom = exp_avg_sq.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(exp_avg_bar, denom, value=-step_size)

        return loss
