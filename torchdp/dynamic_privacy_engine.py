import numpy as np
import torch
from torch import nn
from typing import List

from .privacy_metric import PrivacyMetric, DPOutOfBudgetError
from . import privacy_analysis as tf_privacy
from .privacy_engine import PrivacyEngine
from .utils import stats


class NoiseScheduler(object):
    def __init__(self):
        self.stat = {}

    def __call__(self, t, param_dict):
        return param_dict['initial_noise_multiplier']

    def update_stat(self, stat):
        self.stat = stat


class PredefinedSch(NoiseScheduler):
    def __init__(self, sigmas, sample_rate=1.):
        super().__init__()
        self._sigmas = sigmas
        self.sample_rate = sample_rate

    def __call__(self, t, **param_dict):
        return self._sigmas[int(t*self.sample_rate)]


class TimeDecaySch(NoiseScheduler):
    def __call__(self, t, initial_noise_multiplier=10., k=0.05, **param_dict):
        if param_dict["batch_type"] == "shuffle":
            # t and i_epoch both start from 0.
            # index of current epoch.
            t = np.floor(t * param_dict['sample_rate'])
        return initial_noise_multiplier / (1. + k * t)


class PolyDecaySch(NoiseScheduler):
    def __call__(self, t, initial_noise_multiplier=10., final_noise_multiplier=2., k=3.,
                 period=100., **param_dict):
        if param_dict["batch_type"] == "shuffle":
            # t and i_epoch both start from 0.
            # index of current epoch.
            t = np.floor(t * param_dict['sample_rate'])
        return (initial_noise_multiplier - final_noise_multiplier) * (1 - float(t) / period) ** k \
            + final_noise_multiplier


class ExpDecaySch(NoiseScheduler):
    def __call__(self, t, initial_noise_multiplier=10., k=0.01, **param_dict):
        if param_dict["batch_type"] == "shuffle":
            # t and i_epoch both start from 0.
            # index of current epoch.
            t = np.floor(t * param_dict['sample_rate'])
        return initial_noise_multiplier * np.exp(- k * t)


class StepDecaySch(NoiseScheduler):
    def __call__(self, t, initial_noise_multiplier=10., k=0.6, period=10, **param_dict):
        if param_dict["batch_type"] == "shuffle":
            # t and i_epoch both start from 0.
            # index of current epoch.
            t = np.floor(t * param_dict['sample_rate'])
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
        self.val_epoch = 0

    def __call__(self, t, initial_noise_multiplier=10., k=0.01, period=10, m=5, **param_dict):
        if param_dict["batch_type"] == "shuffle":
            # t and i_epoch both start from 0.
            # index of current epoch.
            self.epoch = np.floor(t * param_dict['sample_rate'])
            # print(f"### epch: {self.epoch} sample rate: {param_dict['sample_rate']}")
        if t == 0 or self.stat["noise_multiplier"] is None:
            # reset at the beginning.
            self.stat["noise_multiplier"] = initial_noise_multiplier
            self.initial_noise_multiplier = initial_noise_multiplier
            self.k = k
            self.m = m
            self.period = period
        return self.stat["noise_multiplier"]

    def update_stat(self, stat):
        assert self.period >= self.m, f"period has to be >= m while period is {self.period}, " \
                                      f"m is {self.m}"
        if self.val_epoch == 0:
            self.stat["val_acc"] = []
        self.stat["val_acc"].append(stat["val_acc"])
        self.val_epoch += 1

        if self.val_epoch > 0 and self.val_epoch % self.period == 0 and self.val_epoch > self.period:
            S_e = np.mean(self.stat["val_acc"][-self.m:])
            S_e1 = np.mean(self.stat["val_acc"][-self.period-self.m:-self.period])
            if S_e - S_e1 < self.delta:
                self.stat["noise_multiplier"] *= self.k


class DynamicPrivacyEngine(PrivacyEngine):
    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        sample_size: int,
        alphas: List[float],
        max_grad_norm: float,
        grad_norm_type: int = 2,
        batch_first: bool = True,
        privacy_budget: PrivacyMetric = None,
        batch_type: str = "shuffle",
        layer_wise_clip: bool = False,
        # e.g., lambda t, param_dict: 10. (return constant).
        dynamic_sch_func: NoiseScheduler = None,
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
                                                   max_grad_norm, grad_norm_type, batch_first,
                                                   privacy_budget=privacy_budget, batch_type=batch_type,
                                                   clip_per_layer=layer_wise_clip)
        self.step_noise_multipliers = []
        self.accumulated_rdp = None
        if dynamic_sch_func is None:
            def dynamic_sch_func(
                t, param_dict): return dyn_fun_param["initial_noise_multiplier"]
        self.dynamic_sch_func = dynamic_sch_func
        self.dyn_fun_param = dyn_fun_param
        self.dyn_fun_param["batch_type"] = self.batch_type
        self.dyn_fun_param["sample_rate"] = self.sample_rate

    def step(self):
        noise_multiplier = self.dynamic_sch_func(
            self.steps, **self.dyn_fun_param)
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
        # record noise multiplier.
        self.step_noise_multipliers += [self.noise_multiplier]
        stats.update(stats.StatType.PRIVACY, 'AllLayers',
                     noise_multiplier=self.noise_multiplier)
        # TODO try to disable this?
        # step_rdp = tf_privacy.compute_rdp(self.sample_rate, noise_multiplier, 1, self.alphas)
        # if self.accumulated_rdp is None:
        #     self.accumulated_rdp = step_rdp
        # else:
        #     self.accumulated_rdp = step_rdp + self.accumulated_rdp

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
            assert np.sum(np.abs(np.array(
                self.step_noise_multipliers[start:end]) - self.step_noise_multipliers[start])) < 1e-3, f"Step noise is not constant at epoch {epoch}."

    def get_privacy_spent(self, target_delta: float):
        if self.accumulated_rdp is None:
            return np.inf, np.nan
        else:
            return tf_privacy.get_privacy_spent(self.alphas, self.accumulated_rdp, target_delta)
