#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import types
import warnings
from typing import List, Union

import torch
from torch import nn

from .privacy_metric import PrivacyMetric, DPOutOfBudgetError
from . import privacy_analysis as tf_privacy
from .dp_model_inspector import DPModelInspector
from .per_sample_gradient_clip import PerSampleGradientClipper
from .utils import clipping


class PrivacyEngine:
    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        sample_size: int,
        alphas: List[float],
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        grad_norm_type: int = 2,
        batch_first: bool = True,
        privacy_budget: PrivacyMetric = None,
        batch_type: str = "shuffle",
        target_delta: float = 1e-8,
        loss_reduction: str = "mean",
        **misc_settings,
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

        self.batch_size = batch_size
        self.sample_rate = batch_size / sample_size
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.grad_norm_type = grad_norm_type
        self.batch_first = batch_first
        self.target_delta = target_delta

        self.privacy_budget = privacy_budget
        # self.n_batches_per_epoch = n_batches_per_epoch  # this will be estimated by `1/self.sample_rate`
        self.batch_type = batch_type
        if privacy_budget is not None:
            self._residual_budget = privacy_budget

        # pyre-fixme[6]: Expected `int` for 1st param but got `None`.
        self._set_seed(None)
        self.validator = DPModelInspector()
        self.clipper: PerSampleGradientClipper = None  # lazy initialization in attach
        self.misc_settings = misc_settings

        self.loss_reduction = loss_reduction

    def detach(self):
        optim = self.optimizer
        optim.privacy_engine = None
        self.clipper.close()
        optim.step = types.MethodType(optim.original_step, optim)
        del optim.virtual_step

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
        norm_clipper = (
            # pyre-fixme[6]: Expected `float` for 1st param but got
            #  `Union[List[float], float]`.
            clipping.ConstantFlatClipper(self.max_grad_norm)
            if not isinstance(self.max_grad_norm, list)
            # pyre-fixme[6]: Expected `List[float]` for 1st param but got
            #  `Union[List[float], float]`.
            else clipping.ConstantPerLayerClipper(self.max_grad_norm)
        )

        if self.misc_settings.get("experimental", False):
            norm_clipper = clipping._Dynamic_Clipper_(
                # pyre-fixme[6]: Expected `List[float]` for 1st param but got
                #  `List[Union[List[float], float]]`.
                [self.max_grad_norm],
                self.misc_settings.get("clip_per_layer", False),
                self.misc_settings.get(
                    "clipping_method", clipping.ClippingMethod.STATIC
                ),
                self.misc_settings.get("ratio", 0.0),
            )

        self.clipper = PerSampleGradientClipper(
            self.module, norm_clipper, self.batch_first
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

        # pyre-fixme[16]: `Optimizer` has no attribute `privacy_engine`.
        optimizer.privacy_engine = self
        # pyre-fixme[16]: `Optimizer` has no attribute `original_step`.
        optimizer.original_step = optimizer.step
        # pyre-fixme[8]: Attribute has type
        #  `BoundMethod[typing.Callable(torch.optim.Optimizer.step)[[Named(self,
        #  torch.optim.Optimizer), Named(closure, typing.Optional[typing.Callable[[],
        #  torch.Tensor]], default)], typing.Optional[torch.Tensor]],
        #  torch.optim.Optimizer]`; used as `MethodType`.
        optimizer.step = types.MethodType(dp_step, optimizer)

        # We add a 'virtual_step' function to the optimizer, which
        # enables the use of virtual batches.
        # By repeatedly computing backward passes and calling virtual_step,
        # we can aggregate the clipped gradient for large batches
        def virtual_step(self):
            self.privacy_engine.virtual_step()

        # pyre-fixme[16]: `Optimizer` has no attribute `virtual_step`.
        optimizer.virtual_step = types.MethodType(virtual_step, optimizer)

        # pyre-fixme[16]: `PrivacyEngine` has no attribute `optimizer`.
        self.optimizer = optimizer  # create a cross reference for detaching

    def get_renyi_divergence(self):
        rdp = torch.tensor(
            tf_privacy.compute_rdp(
                self.sample_rate, self.noise_multiplier, 1, self.alphas
            )
        )
        return rdp

    # pyre-fixme[9]: target_delta has type `float`; used as `None`.
    def get_privacy_spent(self, target_delta: float = None):
        """Compute the exact privacy spent by Moment Accountant.
        TODO Add analytical DP accountant which will be more precise.
        """
        if target_delta is None:
            target_delta = self.target_delta
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
        self.clipper.clip_and_accumulate()
        clip_values, batch_size = self.clipper.pre_step()

        # ensure the clipper consumed the right amount of gradients.
        # In the last batch of a training epoch, we might get a batch that is
        # smaller than others but we should never get a batch that is too large
        if batch_size > self.batch_size:
            raise ValueError(
                f"PrivacyEngine expected a batch of size {self.batch_size} "
                f"but received a batch of size {batch_size}"
            )

        if batch_size < self.batch_size:
            warnings.warn(
                f"PrivacyEngine expected a batch of size {self.batch_size} "
                f"but the last step received a batch of size {batch_size}. "
                "This means that the privacy analysis will be a bit more "
                "pessimistic. You can set `drop_last = True` in your PyTorch "
                "dataloader to avoid this problem completely"
            )

        params = (p for p in self.module.parameters() if p.requires_grad)
        for p, clip_value in zip(params, clip_values):
            noise = self._generate_noise(clip_value, p)
            if self.loss_reduction == "mean":
                noise /= batch_size
            p.grad += noise

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
    
    def virtual_step(self):
        self.clipper.clip_and_accumulate()

    def _generate_noise(self, max_norm, parameter):
        noise_multiplier = self.request_budget()
        if noise_multiplier > 0:
            return torch.normal(
                0,
                noise_multiplier * max_norm,
                parameter.grad.shape,
                device=self.device,
                generator=self.secure_generator,
            )
        return 0.0

    def _set_seed(self, secure_seed: int):
        if secure_seed is not None:
            # pyre-fixme[16]: `PrivacyEngine` has no attribute `secure_seed`.
            self.secure_seed = secure_seed
        else:
            self.secure_seed = int.from_bytes(
                os.urandom(8), byteorder="big", signed=True
            )
        self.secure_generator = (
            torch.random.manual_seed(self.secure_seed)
            if self.device.type == "cpu"
            else torch.cuda.manual_seed(self.secure_seed)
        )
