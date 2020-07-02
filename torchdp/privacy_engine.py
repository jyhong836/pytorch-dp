#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import types
from typing import List
import numpy as np

import torch
from torch import nn

from .privacy_metric import PrivacyMetric, DPOutOfBudgetError
from . import privacy_analysis as tf_privacy
from .dp_model_inspector import DPModelInspector
from . import stats
from .per_sample_gradient_clip import (
    PerSampleGradientClipper,
    __clip_value_calculation_params__ as clipping_method)


class PrivacyEngine:
    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        sample_size: int,
        alphas: List[float],
        noise_multiplier: float,
        max_grad_norm: float,
        grad_norm_type: int = 2,
        batch_dim: int = 0,
        privacy_budget: PrivacyMetric = None,
        batch_type: str = "shuffle",
        layer_wise_clip: bool = False,
    ):
        """

        Args:
            module: PyTorch module.
            batch_size ():
            sample_size ():
            alphas ():
            noise_multiplier ():
            max_grad_norm ():
            grad_norm_type ():
            batch_dim ():
            privacy_budget: Privacy budget. If set, the step will raise DPOutOfBudget except on overflow.
            batch_type ('shuffle'|'random'): Type of batch method. This will result in different privacy counting
                tech.
        """
        self.steps = 0
        self.module = module
        self.alphas = alphas
        self.device = next(module.parameters()).device

        self.sample_rate = batch_size / sample_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = grad_norm_type
        self.batch_dim = batch_dim
        self.privacy_budget = privacy_budget
        # self.n_batches_per_epoch = n_batches_per_epoch  # this will be estimated by `1/self.sample_rate`
        self.batch_type = batch_type
        self.layer_wise_clip = layer_wise_clip
        if privacy_budget is not None:
            self._residual_budget = privacy_budget
            # self._accumulate_cost = privacy_budget.zero()

        self.secure_seed = int.from_bytes(os.urandom(8), byteorder="big", signed=True)
        self.secure_generator = (
            torch.random.manual_seed(self.secure_seed)
            if self.device.type == "cpu"
            else torch.cuda.manual_seed(self.secure_seed)
        )
        self.validator = DPModelInspector()
        self.clipper: PerSampleGradientClipper = None  # lazy initialization in attach

    def detach(self):
        optim = self.optimizer
        optim.privacy_engine = None
        self.clipper.close()
        optim.step = types.MethodType(optim.original_step, optim)

    def attach(self, optimizer: torch.optim.Optimizer):
        """
        Attaches to a `torch.optim.Optimizer` object, and injects itself into
        the optimizer's step.

        To do that, this method does the following:
        1. Validates the model for containing un-attachable layers
        2. Adds a pointer to this object (the PrivacyEngine) inside the optimizer
        3. Moves the original optimizer's `step()` function to `original_step()`
        4. Monkeypatches the optimizer's `step()` function to call `step()` on
        the query engine automatically whenever it would call `step()` for itself
        """

        # Validate the model for not containing un-supported modules.
        self.validator.validate(self.module)
        # only attach if model is validated
        if clipping_method['method'].lower() != 'none':
            print('Warning! Current implementations of dynamic clipping '
                  'are not privacy safe; Caclulated privacy loss is not '
                  'indicative of a proper bound.')
        self.clipper = PerSampleGradientClipper(
            self.module, self.max_grad_norm, self.batch_dim, self.layer_wise_clip
        )

        def dp_step(self, closure=None):
            if closure is not None:
                _closure = closure
                def private_closure():
                    orig_loss = _closure()
                    self.privacy_engine.step()
                    return orig_loss
                closure = private_closure
            else:
                self.privacy_engine.step()
            self.original_step(closure)

        optimizer.privacy_engine = self
        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

        self.optimizer = optimizer  # create a cross reference for detaching

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            tf_privacy.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    def get_privacy_spent(self, target_delta: float):
        """Compute the exact privacy spent by Moment Accountant.
        TODO Add analytical DP accountant which will be more precise.
        """
        rdp = self.get_renyi_divergence() * self.steps
        return tf_privacy.get_privacy_spent(self.alphas, rdp, target_delta)

    def get_privacy_metric_value(self, noise_multiplier):
        cost = self._residual_budget.from_sigma(noise_multiplier)
        # FIXME: The batch type should be update when we fix the shuffling issue.
        cost = cost.amp_by_sampling(self.sample_rate, batch_type=self.batch_type)
        return cost

    def request_budget(self):
        """Try to request an amount of budget based on current `noise_multiplier`. Call this
        before apply the noise to protect privacy.

        Returns:
            noise_multiplier: if request is approved.

        Raises:
            DPOutOfBudgetError: If residual budget is not enough
        """
        if self.privacy_budget is not None:
            cost = self.get_privacy_metric_value(self.noise_multiplier)
            if self._residual_budget < cost:
                # reject
                raise DPOutOfBudgetError(f"Request budget ({cost:.3g}) > "
                                         f"residual ({self._residual_budget:.3g}) at step {self.steps}.")
            else:
                # self._accumulate_cost = self._accumulate_cost.compose([self._accumulate_cost, cost])
                # FIXME The substraction may only be defined for some metric but compose is for all qualified metrics.
                self._residual_budget = self._residual_budget - cost
                # diff = self.privacy_budget.compose([self._accumulate_cost, self._residual_budget]) - self.privacy_budget
                # assert(abs(diff.rho) < 1e-15)
                # print(self.privacy_budget.from_sigma(self.noise_multiplier))
                cost__ = self.privacy_budget.from_sigma(self.noise_multiplier).amp_by_sampling(self.sample_rate, batch_type=self.batch_type)
                # print(cost, cost__)
        return self.noise_multiplier

    def step(self):
        self.steps += 1
        max_norm = self.clipper.step()
        noise_multiplier = self.request_budget()
        for p in self.module.parameters():
            if p.requires_grad and self.noise_multiplier > 0:
                noise = torch.normal(
                    0,
                    noise_multiplier * max_norm,
                    p.grad.shape,
                    device=self.device,
                    generator=self.secure_generator,
                )
                p.grad += noise / self.clipper.batch_size

    def to(self, device):
        self.device = device
        return self

    def get_budget_usage(self, str_only=True):
        if self.privacy_budget is None:
            return "" if str_only else None
        if str_only:
            return f"residual budget = {self._residual_budget:3g}/{self.privacy_budget:3g}"
        else:
            return self._residual_budget


class NoiseScheduler(object):
    def __init__(self):
        self.stat = {}

    def __call__(self, t, param_dict):
        return param_dict['initial_noise_multiplier']

    def update_stat(self, stat):
        self.stat = stat


class PredefinedSch(NoiseScheduler):
    def __init__(self, sigmas):
        super().__init__()
        self._sigmas = sigmas

    def __call__(self, t, **param_dict):
        return self._sigmas[t]


class ExpDecaySch(NoiseScheduler):
    def __call__(self, t, initial_noise_multiplier=10., k=0.01, **param_dict):
        if param_dict["batch_type"] == "shuffle":
            # t and i_epoch both start from 0.
            t = np.floor(t * param_dict['sample_rate'])  # index of current epoch.
        return initial_noise_multiplier * np.exp(- k * t)


class StepDecaySch(NoiseScheduler):
    def __call__(self, t, initial_noise_multiplier=10., k=0.6, period=10, **param_dict):
        if param_dict["batch_type"] == "shuffle":
            # t and i_epoch both start from 0.
            t = np.floor(t * param_dict['sample_rate'])  # index of current epoch.
        return initial_noise_multiplier * (k ** (t // period))


class ValDecaySch(NoiseScheduler):
    def __init__(self):
        super().__init__()
        self.stat["noise_multiplier"] = None
        self.stat["val_acc"] = None
        self.initial_noise_multiplier = None
        self.k = None
        self.delta = 0.01
        self.period = 10
        self.epoch = 0

    def __call__(self, t, initial_noise_multiplier=10., k=0.01, period=10, **param_dict):
        if self.stat["noise_multiplier"] is None:
            self.stat["noise_multiplier"] = initial_noise_multiplier
            self.initial_noise_multiplier = initial_noise_multiplier
            self.k = k
            self.period = period
        return self.stat["noise_multiplier"]

    def update_stat(self, stat):
        if self.epoch % self.period == 0:
            if self.stat["val_acc"] is None:
                self.stat["val_acc"] = stat["val_acc"]
            else:
                if stat["val_acc"] - self.stat["val_acc"] < self.delta:
                    self.stat["noise_multiplier"] *= (1 - self.k)
                self.stat["val_acc"] = stat["val_acc"]
        self.epoch += 1


class DynamicPrivacyEngine(PrivacyEngine):
    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        sample_size: int,
        alphas: List[float],
        max_grad_norm: float,
        grad_norm_type: int = 2,
        batch_dim: int = 0,
        privacy_budget: PrivacyMetric = None,
        batch_type: str = "shuffle",
        layer_wise_clip: bool = False,
        dynamic_sch_func: NoiseScheduler = None,  # e.g., lambda t, param_dict: 10. (return constant).
        **dyn_fun_param
    ):
        """
        Args:
            dyn_fun_param: Dict of parameters for dynamic_sch_func. Possible params which depends on the sch_func:
                dynamic_interval: The step interval for updating noise multiplier. Use this to make epoch-wise
                    schedule. For example, set it to be the number of batch in one epoch.
                initial_noise_multiplier: The initial noise.
        """
        initial_noise_multiplier = dyn_fun_param["initial_noise_multiplier"]
        super(DynamicPrivacyEngine, self).__init__(module, batch_size, sample_size, alphas, initial_noise_multiplier,
                                                   max_grad_norm, grad_norm_type, batch_dim,
                                                   privacy_budget=privacy_budget, batch_type=batch_type,
                                                   layer_wise_clip=layer_wise_clip)
        self.step_noise_multipliers = []
        self.accumulated_rdp = None
        if dynamic_sch_func is None:
            def dynamic_sch_func(t, param_dict): return dyn_fun_param["initial_noise_multiplier"]
        self.dynamic_sch_func = dynamic_sch_func
        self.dyn_fun_param = dyn_fun_param
        self.dyn_fun_param["batch_type"] = self.batch_type
        self.dyn_fun_param["sample_rate"] = self.sample_rate

    def step(self):
        noise_multiplier = self.dynamic_sch_func(self.steps, **self.dyn_fun_param)
        old_noise_multiplier = self.noise_multiplier
        self.noise_multiplier = noise_multiplier
        try:
            # Try to apply the noise multiplier.
            super().step()
            self.record_step_noise_multiplier(self.noise_multiplier)
        except DPOutOfBudgetError as e:
            self.noise_multiplier = old_noise_multiplier
            raise e

    def record_step_noise_multiplier(self, noise_multiplier):
        self.step_noise_multipliers += [self.noise_multiplier]  # record noise multiplier.
        stats.update(stats.StatType.PRIVACY, 'AllLayers', noise_multiplier=self.noise_multiplier)
        step_rdp = tf_privacy.compute_rdp(self.sample_rate, noise_multiplier, 1, self.alphas)
        if self.accumulated_rdp is None:
            self.accumulated_rdp = step_rdp
        else:
            self.accumulated_rdp = step_rdp + self.accumulated_rdp

    def get_renyi_divergence(self):
        # TODO use this to verify the dynamic.
        # self.validate_noise_dynamic()
        step_rdps = torch.tensor(
            [tf_privacy.compute_rdp(
                self.sample_rate, noise_multiplier, 1, self.alphas
            ) for noise_multiplier in self.step_noise_multipliers]
        )
        # print(step_rdps.shape)  # [n_batch, n_alpha]
        return step_rdps

    def validate_noise_dynamic(self):
        """Check the step noise multipliers in current epoch. This will also check if the
        batch round the number of samples by examining the sample_rate."""
        # FIXME this will be very inefficient to do in steps.
        if self.batch_type == "shuffle":
            # TODO check epoch-wise constant noise.
            n_batchs_per_epoch = 1/self.sample_rate
            assert n_batchs_per_epoch.is_integer(), "Number of batch is not integer. Try to " \
                                                    "select batch size to round the data size."
            n_batchs_per_epoch = int(n_batchs_per_epoch)
            T = len(self.step_noise_multipliers) - 1
            if T + 1 < n_batchs_per_epoch:
                return
            epoch = int(np.floor(T * self.sample_rate))
            start = epoch*n_batchs_per_epoch
            end = np.minimum((epoch+1)*n_batchs_per_epoch, T)
            # print(start, len(self.step_noise_multipliers), T)
            assert np.sum(np.abs(np.array(self.step_noise_multipliers[start:end]) - self.step_noise_multipliers[start])) < 1e-3, f"Step noise is not constant at epoch {epoch}."

    def get_privacy_spent(self, target_delta: float):
        if self.accumulated_rdp is None:
            return np.inf, np.nan
        else:
            return tf_privacy.get_privacy_spent(self.alphas, self.accumulated_rdp, target_delta)
