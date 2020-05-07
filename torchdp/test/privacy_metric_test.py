import unittest
from torchdp import privacy_metric
from .. import privacy_analysis as tf_privacy


class test_CDP(unittest.TestCase):
    def test_dp_zcdp(self):
        # config: uniform schedule
        sigma = 8.
        T = 100
        sample_rate = 1.
        target_delta = 1e-8
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        zcdp = privacy_metric.zCDP.from_sigma(sigma=sigma)
        zcdp = privacy_metric.zCDP(rho=zcdp.rho * T)

        rdps = tf_privacy.compute_rdp(sample_rate, sigma, 1, alphas)
        rdp = rdps * T
        eps, _ = tf_privacy.get_privacy_spent(alphas, rdp, target_delta)
        dp = privacy_metric.DP(eps, target_delta)

        print(zcdp.to_dp(target_delta).eps, dp.eps)
        self.assertAlmostEqual(zcdp.to_dp(target_delta).eps, dp.eps, places=3)

    def test_dp_ctcdp(self):
        # config: uniform schedule
        sigma = 8.
        T = 100
        sample_rate = 1.
        target_delta = 1e-8
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        rdps = tf_privacy.compute_rdp(sample_rate, sigma, 1, alphas)
        rdp = rdps * T
        eps, _ = tf_privacy.get_privacy_spent(alphas, rdp, target_delta)
        dp = privacy_metric.DP(eps, target_delta)

        # only test forward and backward
        ctcdp = privacy_metric.ctCDP.from_dp(dp)
        ctcdp = privacy_metric.ctCDP(rho=ctcdp.rho)

        print(ctcdp.to_dp(target_delta).eps, dp.eps)
        self.assertAlmostEqual(ctcdp.to_dp(target_delta).eps, dp.eps, places=3)
