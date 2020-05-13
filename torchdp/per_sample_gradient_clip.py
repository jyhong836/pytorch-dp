#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
In order to apply the Gaussian mechanism to the gradient computation, it is
necessary to bound its sensitivity.
This can be achieved via **per-sample gradient clipping** (in short,
*grad_sample clip*).
Normally if you have a matrix of parameters of size [m, n], the size of the
gradients will match it. This means that they get aggregated over the batch.
Here, we will keep them per-example ie we will have a tensor of size [b_sz, m, n].

grad_sample clip has to be achieved under the following constraints:

1. The norm of the grad_sample of the loss wrt all model parameters has
to be clipped so that if they were to be put in a single vector together, the
total norm will be at most C.
Or in code, let `T = torch.cat([p.grad_sample.flatten() for p in model.parameters()])`.
T will have size `[B, N_TOTAL_PARAMS]`. The total L2 norm of each row of T
cannot be greater than C.
2. This clipping should not backpropagate. This means that clipping layer i+1
should not affect computing the gradient of layer i. To make sure this doesn't
happen, we will first compute the grad_sample of all layers
**without clipping**. In a second pass, we will go back to the per-sample
gradients, clip them, and accumulate them in `.grad`
(thus replacing the "real" gradients). Note: there is only a single .backward()
call as the second pass just works on top of the store grad_sample.
"""

import torch
import numpy as np

from . import autograd_grad_sample
from . import stats


__clip_value_calculation_params__ = {'method' : 'none',
                                     'factor' : -1,
                                     'percentile' : .875}


def _calc_thresh(data : torch.Tensor,
                   method : str = 'none',
                   current_max : float = -1,
                   factor : float = -1,
                   percentile : float = .875) -> float:
    """
    Calculates the clipping threshold by looking at the layer norms
    of each example. Three methods are supported: static threshold,
    threshold calculated based on mean and variance of the norms, and
    threshold calculated based on percentile values of the norms.
    """
    method = method.lower()
    if method == 'none':
        return current_max
    elif method == 'mean_var':
        return max(data.min().item(),
                   data.mean().item() + factor * data.std().item() + 1e-8)
    elif method == 'pvalue':
        cut = max(1, int(data.numel() * (1 - percentile)))
        return torch.kthvalue(data, cut)[0].item()


def clip_per_sample_grad_norm_(model, max_norm) -> float:
    r"""Clips the grad_sample stored in .grad_sample by computing a per-sample
    norm clip factor, using it to rescale each sample's gradient in
    .grad_sample to norm clip, then averaging them back into .grad.

    The gradients of the model's parameters are modified in-place.

    We assume the batch size is the first dimension.

    Arguments:
        tensor (Tensor): a single Tensor whose norm will be normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        New total norm of the tensor.
    """
    per_sample_norm = get_total_per_sample_grad_norm(model)
    max_norm = _calc_thresh(per_sample_norm, current_max=float(max_norm),
                            **__clip_value_calculation_params__)
    # Each sample gets clipped independently. This is a tensor of size B
    per_sample_clip_factor = max_norm / (per_sample_norm + 1e-6)

    # We are *clipping* the gradient, so if the factor is ever >1 we set it to 1
    per_sample_clip_factor = per_sample_clip_factor.clamp(max=1.0)
    b_sz = len(per_sample_clip_factor)

    # We recompute .grad from .grad_sample by simply averaging it over the B dim
    sign_switched = 0
    total_num = 0
    for p in model.parameters():
        if p.requires_grad:
            pre_clip_pos = p.grad_sample.mean(0) > 0
            p.grad = torch.einsum("i,i...", per_sample_clip_factor, p.grad_sample) / b_sz
            post_clip_pos = p.grad > 0
            sign_switched += (pre_clip_pos ^ post_clip_pos).sum()
            total_num += post_clip_pos.numel()
    sign_switched = float(sign_switched) / total_num
    stats.update(stats.StatType.CLIPPING, 'AllLayers',
                 clip=max_norm,
                 max=per_sample_norm.max(),
                 mean=per_sample_norm.mean(),
                 median=per_sample_norm.median(),
                 percent=(per_sample_norm > max_norm).to(dtype=torch.float64).mean(),
                 switch=sign_switched)
    return max_norm


def clip_per_sample_layer_grad_norm_(model, max_norm) -> float:
    r"""Clips the grad_sample stored in .grad_sample by computing a per-sample
    norm clip factor, using it to rescale each sample's gradient in
    .grad_sample to norm clip, then averaging them back into .grad.

    The gradients of the model's parameters are modified in-place.

    We assume the batch size is the first dimension.

    Arguments:
        tensor (Tensor): a single Tensor whose norm will be normalized
        max_norm (float or int): max norm of the gradients of each layer

    Returns:
        New total norm of the tensor. The norm of all layers is computed by
        torch.norm([max norm of layer 1, max norm of layer 2,... ])
    """
    per_sample_layer_norms = get_total_per_sample_grad_norm(model, reduce_layer=False)
    n_layers = per_sample_layer_norms.shape[1]  # num of layers that requires grad.
    max_norm = [_calc_thresh(per_sample_layer_norms[:, i], current_max=float(max_norm),
                             **__clip_value_calculation_params__) for i in range(n_layers)]
    # Each sample gets clipped independently. This is a tensor of size B
    per_sample_clip_factor = [(max_norm[i] / (per_sample_layer_norms[:, i] + 1e-6)).clamp(max=1.0)
                              for i in range(n_layers)]
    total_max_norm = np.linalg.norm(max_norm)
    # total_max_norm = max_norm[0]  # TODO this may be wrong

    # We are *clipping* the gradient, so if the factor is ever >1 we set it to 1
    # per_sample_clip_factor = per_sample_clip_factor.clamp(max=1.0)
    b_sz = len(per_sample_clip_factor[0])

    # We recompute .grad from .grad_sample by simply averaging it over the B dim
    sign_switched = 0
    total_num = 0
    i_layer = 0  # index of layer that requires grad.
    for p_name, p in model.named_parameters():
        if p.requires_grad:
            if p_name.endswith("bias"):
                # print("!!!!")
                i_layer -= 1
            # print(f"{p_name}")
            pre_clip_pos = p.grad_sample.mean(0) > 0
            p.grad = torch.einsum("i,i...", per_sample_clip_factor[i_layer], p.grad_sample) / b_sz
            post_clip_pos = p.grad > 0
            sign_switched += (pre_clip_pos ^ post_clip_pos).sum()
            total_num += post_clip_pos.numel()
            i_layer += 1
    sign_switched = float(sign_switched) / total_num
    stats.update(stats.StatType.CLIPPING, 'AllLayers',
                 # clip=max_norm,  # this is a list of max_norm for all layers.
                 # max=per_sample_norm.max(),
                 # mean=per_sample_norm.mean(),
                 # median=per_sample_norm.median(),
                 # percent=(per_sample_norm > max_norm).to(dtype=torch.float64).mean(),
                 switch=sign_switched)
    return total_max_norm


def get_per_sample_norm(t, name, stat):
    aggregation_dims = [i for i in range(1, len(t.shape))]  # All dims except the first
    t_squared = t * t  # elementwise
    batch_norms = torch.sqrt(t_squared.sum(dim=aggregation_dims))
    normalized_per_coordinate_value = t.abs().sum(dim=aggregation_dims) / t[0].numel()
    stat[f'{name}:max'] = normalized_per_coordinate_value.max()
    stat[f'{name}:mean'] = normalized_per_coordinate_value.mean()
    stat[f'{name}:unnormalized_max'] = batch_norms.max()
    return batch_norms


def get_total_per_sample_grad_norm(model, reduce_layer=True):
    stat = {}
    if not reduce_layer:
        all_layers_norms = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                norms = get_per_sample_norm(p.grad_sample, name, stat)
                if name.endswith("bias"):
                    weight_norm = all_layers_norms.pop()
                    norms = torch.norm(torch.stack([norms, weight_norm], dim=-1), dim=1)
                all_layers_norms += [norms]
        all_layers_norms = torch.stack(all_layers_norms, dim=-1)
    else:
        all_layers_norms = torch.stack(
            [get_per_sample_norm(p.grad_sample, name, stat)\
                for name, p in model.named_parameters() if p.requires_grad], dim=-1
        )
    stats.update(stats.StatType.CLIPPING, 'IndividualLayers', **stat)
    return all_layers_norms.norm(2, dim=1) if reduce_layer else all_layers_norms


class PerSampleGradientClipper:
    def __init__(self, module, max_norm, batch_dim=0, layer_wise=False):
        """
        Attaches to a module, and clips all grad_sample in the backward
        pass. It then puts them in each parameter's .grad.
        """
        self.module = module
        autograd_grad_sample.add_hooks(self.module)
        self.max_norm = max_norm
        self.hooks_attached = True
        self.batch_dim = batch_dim
        self.layer_wise = layer_wise

    def __del__(self):
        self.close()

    def close(self):
        if self.hooks_attached:  # do not close twice
            autograd_grad_sample.remove_hooks(self.module)
        self.hooks_attached = False

    def __repr__(self):
        return f"PerSampleGradientClipModuleHook on {self.module}"

    def step(self):
        autograd_grad_sample.compute_grad_sample(self.module, batch_dim=self.batch_dim)

        # The first dim of param.grad_sample is b_sz for every param.
        # To look up what that value is, we just pick one
        self.batch_size = next(
            p.grad_sample.shape[0] for p in self.module.parameters() if p.requires_grad
        )

        if self.layer_wise:
            max_norm = clip_per_sample_layer_grad_norm_(self.module, self.max_norm)
        else:
            max_norm = clip_per_sample_grad_norm_(self.module, self.max_norm)
        autograd_grad_sample.clear_backprops(self.module)
        return max_norm
