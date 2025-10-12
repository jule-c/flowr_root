from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch.optim.optimizer import Optimizer

# functions


def exists(val):
    return val is not None


# tensor helpers


def log(t, eps=1e-20):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


def spectral_entropy_reg_loss_hook(optimizer, *args, **kwargs):
    loss = torch.tensor(0.0).requires_grad_()

    for param_group in optimizer.param_groups:
        if not param_group["add_spectral_entropy_reg"]:
            continue

        weight = param_group["spectral_entropy_reg_weight"]
        use_svd_lowrank = param_group["use_svd_lowrank"]
        svd_lowrank_kwargs = param_group["svd_lowrank_kwargs"]

        for param in param_group["params"]:
            if param.ndim < 2:
                continue

            *_, row, col = param.shape
            reshaped_param = param.reshape(-1, row, col)

            if not use_svd_lowrank:
                singular_values = torch.linalg.svdvals(reshaped_param)
            else:
                _, singular_values, _ = torch.svd_lowrank(
                    reshaped_param, **svd_lowrank_kwargs
                )

            spectral_prob = singular_values.softmax(dim=-1)
            spectral_entropy = entropy(spectral_prob).sum()
            loss = loss + spectral_entropy

    (loss * weight).backward()


# class


class GrokFastAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay=0.01,
        eps=1e-8,
        regen_reg_rate=0.0,
        grokfast=True,
        grokfast_alpha=0.9,
        grokfast_lamb=1.0,
        grokfast_after_step=0,
        normalize_lr=True,
        add_spectral_entropy_reg=False,
        spectral_entropy_reg_weight=0.1,
        use_svd_lowrank=False,
        svd_lowrank_kwargs: dict = dict(),
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        assert weight_decay >= 0.0
        assert regen_reg_rate >= 0.0
        assert eps > 0.0
        assert not (
            weight_decay > 0.0 and regen_reg_rate > 0.0
        ), "weight decay and regenerative regularization cannot be used together"

        # in order for fair comparison
        # reduce the learning rate by a factor of (1 + grokfast_lamb)

        if grokfast and normalize_lr:
            lr /= 1.0 + grokfast_lamb

        self._init_lr = lr

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            regen_reg_rate=regen_reg_rate,
            grokfast=grokfast,
            grokfast_alpha=grokfast_alpha,
            grokfast_lamb=grokfast_lamb,
            grokfast_after_step=grokfast_after_step,
            add_spectral_entropy_reg=add_spectral_entropy_reg,
            spectral_entropy_reg_weight=spectral_entropy_reg_weight,
            use_svd_lowrank=use_svd_lowrank,
            svd_lowrank_kwargs=svd_lowrank_kwargs,
        )

        super().__init__(params, defaults)

        # maybe spectral entropy reg
        # https://openreview.net/forum?id=07N9jCfIE4

        if not add_spectral_entropy_reg:
            return

        self.register_step_pre_hook(partial(spectral_entropy_reg_loss_hook, self))

    def turn_on_grokfast(self):
        for group in self.param_groups:
            group["grokfast"] = True

    def turn_off_grokfast(self):
        for group in self.param_groups:
            group["grokfast"] = False

    def clear_grokfast_state(self):
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):
                state = self.state[p]
                state.pop("grok_exp_avg", None)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group["params"]):

                (
                    grad,
                    lr,
                    wd,
                    regen_rate,
                    beta1,
                    beta2,
                    eps,
                    grokfast,
                    grokfast_after_step,
                    alpha,
                    lamb,
                    state,
                    init_lr,
                ) = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    group["regen_reg_rate"],
                    *group["betas"],
                    group["eps"],
                    group["grokfast"],
                    group["grokfast_after_step"],
                    group["grokfast_alpha"],
                    group["grokfast_lamb"],
                    self.state[p],
                    self._init_lr,
                )

                # decoupled weight decay

                if wd > 0.0:
                    p.mul_(1.0 - lr / init_lr * wd)

                # regenerative regularization - ICLR 2024
                # https://openreview.net/forum?id=lyoOWX0e0O

                if regen_rate > 0.0 and "param_init" in state:
                    param_init = state["param_init"]

                    p.lerp_(param_init, lr / init_lr * regen_rate)

                # init state if needed

                if len(state) == 0:
                    state["steps"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                    if regen_rate > 0.0:
                        state["param_init"] = p.data.clone()

                # get some of the states

                exp_avg, exp_avg_sq, steps = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["steps"],
                )

                steps += 1

                # should grokfast

                should_grokfast = grokfast and steps > grokfast_after_step and lamb > 0

                # take care of grok fast if turned on

                if should_grokfast:

                    if "grok_exp_avg" not in state:
                        # maintain an ema of the grad
                        # for amplifying slow gradients, as paper claims it accelerates generalization

                        state["grok_exp_avg"] = grad.clone()

                    grok_exp_avg = state["grok_exp_avg"]

                    # update grok exp avg

                    grok_exp_avg.lerp_(grad, 1.0 - alpha)

                    # appendix C - line 24 of https://arxiv.org/html/2405.20233v2

                    grad.add_(grok_exp_avg, alpha=lamb)

                # bias corrections

                bias_correct1 = 1.0 - beta1**steps
                bias_correct2 = 1.0 - beta2**steps

                # decay running averages

                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.lerp_(grad * grad, 1.0 - beta2)

                # adam

                update = (
                    -lr
                    * (exp_avg / bias_correct1)
                    / (exp_avg_sq / bias_correct2).sqrt().clamp(min=eps)
                )

                p.add_(update)

                # increment steps

                state["steps"] = steps

        return loss
