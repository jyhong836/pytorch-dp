#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import types
from typing import List

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
        if privacy_budget is not None:
            self._residual_budget = privacy_budget

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
            self.module, self.max_grad_norm, self.batch_dim
        )

        def dp_step(self, closure=None):
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
        cost = cost.amp_by_sampling(self.sample_rate, batch_type="random")
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
                self._residual_budget = self._residual_budget - cost
        return self.noise_multiplier

    def step(self):
        self.steps += 1
        max_norm = self.clipper.step()
        for p in self.module.parameters():
            if p.requires_grad and self.noise_multiplier > 0:
                noise_multiplier = self.request_budget()
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
        dynamic_sch_func=None,  # e.g., lambda t, param_dict: 10. (return constant).
        **dyn_fun_param
    ):
        """The dict `dyn_fun_param` will be passed to dynamic_sch_func as the 2nd param. It should at least include
        key `initial_noise_multiplier`."""
        initial_noise_multiplier = dyn_fun_param["initial_noise_multiplier"]
        super(DynamicPrivacyEngine, self).__init__(module, batch_size, sample_size, alphas, initial_noise_multiplier,
                                                   max_grad_norm, grad_norm_type, batch_dim,
                                                   privacy_budget=privacy_budget)
        self.step_noise_multipliers = []
        if dynamic_sch_func is None:
            def dynamic_sch_func(t, param_dict): return dyn_fun_param["initial_noise_multiplier"]
        self.dynamic_sch_func = dynamic_sch_func
        self.dyn_fun_param = dyn_fun_param

    def step(self):
        self.noise_multiplier = self.dynamic_sch_func(self.steps, self.dyn_fun_param)
        self.step_noise_multipliers += [self.noise_multiplier]  # record noise multiplier.
        stats.update(stats.StatType.PRIVACY, 'AllLayers', noise_multiplier=self.noise_multiplier)
        super().step()

    def get_renyi_divergence(self):
        step_rdps = torch.tensor(
            [tf_privacy.compute_rdp(
                self.sample_rate, noise_multiplier, 1, self.alphas
            ) for noise_multiplier in self.step_noise_multipliers]
        )
        # print(step_rdps.shape)  # [n_batch, n_alpha]
        return step_rdps

    def get_privacy_spent(self, target_delta: float):
        rdp = torch.sum(self.get_renyi_divergence(), dim=0)
        return tf_privacy.get_privacy_spent(self.alphas, rdp, target_delta)
