import unittest
from torchdp import privacy_metric
from .. import privacy_analysis as tf_privacy


class test_CDP(unittest.TestCase):
    def setUp(self) -> None:
        # config: uniform schedule
        self.sigma = 8.
        self.T = 100
        self.sample_rate = 1.
        self.target_delta = 1e-8
        self.alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def test_dp_zcdp(self):
        # zCDP from noise
        zcdp = privacy_metric.zCDP.from_sigma(sigma=self.sigma)
        zcdp = privacy_metric.zCDP(rho=zcdp.rho * self.T)

        # MA DP from noise
        rdps = tf_privacy.compute_rdp(self.sample_rate, self.sigma, 1, self.alphas)
        rdp = rdps * self.T
        eps, _ = tf_privacy.get_privacy_spent(self.alphas, rdp, self.target_delta)
        dp = privacy_metric.DP(eps, self.target_delta)

        self.assertAlmostEqual(zcdp.to_dp(self.target_delta).eps, dp.eps, places=3)

        # DP to zCDP
        zcdp = privacy_metric.zCDP.from_dp(dp)
        self.assertAlmostEqual(zcdp.to_dp(self.target_delta).eps, dp.eps, places=3)

    def test_dp_ctcdp(self):
        # MA DP from noise
        rdps = tf_privacy.compute_rdp(self.sample_rate, self.sigma, 1, self.alphas)
        rdp = rdps * self.T
        eps, _ = tf_privacy.get_privacy_spent(self.alphas, rdp, self.target_delta)
        dp = privacy_metric.DP(eps, self.target_delta)

        # only test forward and backward
        ctcdp = privacy_metric.ctCDP.from_dp(dp)
        ctcdp = privacy_metric.ctCDP(rho=ctcdp.rho)

        self.assertAlmostEqual(ctcdp.to_dp(self.target_delta).eps, dp.eps, places=3)

    def test_dp_ctcdp_smp(self):
        self.sample_rate = 0.01

        # MA DP from noise
        rdps = tf_privacy.compute_rdp(self.sample_rate, self.sigma, 1, self.alphas)
        rdp = rdps * self.T / self.sample_rate  # assume T represents # of epochs.
        eps, _ = tf_privacy.get_privacy_spent(self.alphas, rdp, self.target_delta)
        dp = privacy_metric.DP(eps, self.target_delta)

        # test forward and backward
        ctcdp = privacy_metric.ctCDP.from_dp(dp)
        # step cost
        step_ctcdp = privacy_metric.ctCDP(rho=ctcdp.rho/self.T * self.sample_rate).deamp_by_sampling(self.sample_rate)
        ctcdp_sigma = ctcdp.from_sigma(self.sigma).amp_by_sampling(self.sample_rate)
        est_T = ctcdp.rho / ctcdp_sigma.rho * self.sample_rate
        print(self.sigma, est_T)
        res1 = ctcdp - privacy_metric.ctCDP(rho=ctcdp_sigma.rho*1000)
        res2 = ctcdp
        print(ctcdp_sigma)
        for _ in range(1000):
            res2 -= ctcdp_sigma
        print(res1, res2)
        self.assertAlmostEqual(res1.rho, res2.rho)

        self.assertAlmostEqual(ctcdp.to_dp(self.target_delta).eps, dp.eps, places=3)
        self.assertAlmostEqual(step_ctcdp.to_sigma()/10., self.sigma/10., places=1, msg="Step noise multiplier should "
                                                                                        "be approximately equal.")
