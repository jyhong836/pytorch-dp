import unittest
from torchdp import privacy_metric
from .. import privacy_analysis as tf_privacy


class test_CDP(unittest.TestCase):
    # Test uniform schedule
    def setUp(self) -> None:
        # config: uniform schedule
        # example for MNIST + Pipeline(PCA, Scale, MLP1)
        self.sigma = 8.
        self.epochs = 100
        self.sample_rate = 0.01
        self.target_delta = 1e-8
        self.alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def test_dp_zcdp(self):
        sample_rate = self.sample_rate
        batch_type = "shuffle"  # For DP MA, only shuffle is supported.

        # zCDP from noise
        zcdp = privacy_metric.zCDP.from_sigma(sigma=self.sigma)
        zcdp = zcdp.amp_by_sampling(sample_rate, batch_type=batch_type)
        zcdp = privacy_metric.zCDP(rho=zcdp.rho * self.epochs / sample_rate)

        # MA DP from noise
        rdps = tf_privacy.compute_rdp(1., self.sigma, 1, self.alphas)
        rdp = rdps * self.epochs
        eps, _ = tf_privacy.get_privacy_spent(self.alphas, rdp, self.target_delta)
        dp = privacy_metric.DP(eps, self.target_delta)

        self.assertAlmostEqual(zcdp.from_dp(dp).rho, zcdp.rho, places=3)
        self.assertAlmostEqual(zcdp.to_dp(self.target_delta).eps, dp.eps, places=3)

        # DP to zCDP
        zcdp = privacy_metric.zCDP.from_dp(dp)
        self.assertAlmostEqual(zcdp.to_dp(self.target_delta).eps, dp.eps, places=3)

    def test_dp_ctcdp(self):
        sample_rate = 1.  # full batch
        batch_type = "shuffle"  # For DP MA, only shuffle is supported.

        # MA DP from noise
        rdps = tf_privacy.compute_rdp(sample_rate, self.sigma, 1, self.alphas)
        rdp = rdps * self.epochs
        eps, _ = tf_privacy.get_privacy_spent(self.alphas, rdp, self.target_delta)
        dp = privacy_metric.DP(eps, self.target_delta)

        # only test forward and backward
        ctcdp = privacy_metric.ctCDP.from_dp(dp)
        ctcdp = privacy_metric.ctCDP(rho=ctcdp.rho)

        self.assertAlmostEqual(ctcdp.to_dp(self.target_delta).eps, dp.eps, places=3)

    def test_dp_ctcdp_smp(self):
        batch_type = "random"

        # MA DP from noise
        rdps = tf_privacy.compute_rdp(self.sample_rate, self.sigma, 1, self.alphas)
        rdp = rdps * self.epochs / self.sample_rate
        eps, _ = tf_privacy.get_privacy_spent(self.alphas, rdp, self.target_delta)
        dp = privacy_metric.DP(eps, self.target_delta)

        # test forward and backward
        ctcdp = privacy_metric.ctCDP.from_dp(dp)
        print(dp, ctcdp)
        # step cost
        step_ctcdp = privacy_metric.ctCDP(rho=ctcdp.rho / self.epochs * self.sample_rate).deamp_by_sampling(self.sample_rate, batch_type=batch_type)
        ctcdp_sigma = ctcdp.from_sigma(self.sigma).amp_by_sampling(self.sample_rate,
                                                                   batch_type=batch_type)
        est_T = ctcdp.rho / ctcdp_sigma.rho * self.sample_rate
        self.assertAlmostEqual(est_T / 100, self.epochs / 100, places=1)
        # print(self.sigma, est_T)
        res1 = ctcdp - privacy_metric.ctCDP(rho=ctcdp_sigma.rho*1000)
        res2 = ctcdp
        # print(ctcdp_sigma)
        for _ in range(1000):
            res2 -= ctcdp_sigma
        # print(res1, res2)
        self.assertAlmostEqual(res1.rho, res2.rho)

        self.assertAlmostEqual(ctcdp.to_dp(self.target_delta).eps, dp.eps, places=3)
        self.assertAlmostEqual(step_ctcdp.to_sigma()/10., self.sigma/10., places=1,
                               msg="Step noise multiplier should be approximately equal.")

    # TODO test dynamic schedule

    def compose_by_ma(self, sample_rate, T):
        # MA DP from noise
        rdps = tf_privacy.compute_rdp(sample_rate, self.sigma, 1, self.alphas)
        rdp = rdps * T
        eps, alpha = tf_privacy.get_privacy_spent(self.alphas, rdp, self.target_delta)
        dp = privacy_metric.DP(eps, self.target_delta)
        return dp

    def compose_by_cdp(self, metric_type, sample_rate, T, batch_type):
        """metric_type: zCDP or ctCDP."""
        # set the global constraint omega first.
        cdp = metric_type.from_sigma(self.sigma).amp_by_sampling(sample_rate, batch_type=batch_type)
        if batch_type == "shuffle":
            assert (T*sample_rate).is_integer()
        cdp = metric_type(rho=cdp.rho * T)
        return cdp

    def test_RS_sampling_valid(self):
        """Busing uniform schedule, check if the sampling amplification valid, i.e.,
        the cost after amplification is larger than the standard cost (DP) after composition.
        Fail if the metric compose cost is lower than MA or much higher than MA.
        """
        sample_rate = self.sample_rate
        batchs_per_epoch = int(1 / self.sample_rate)
        batch_type = "random"
        T = self.epochs * batchs_per_epoch

        # MA DP from noise
        dp_total_budget = self.compose_by_ma(sample_rate, T)

        # ctCDP
        # set the global constraint omega first.
        ctcdp_total_budget = privacy_metric.ctCDP.from_dp(dp_total_budget)
        ctcdp_total_cost = self.compose_by_cdp(ctcdp_total_budget, # type(ctcdp_total_budget),
                                                 sample_rate, T, batch_type)

        # budget (ctCDP) == budget (DP)
        self.assertAlmostEqual(ctcdp_total_budget.to_dp(self.target_delta).eps, dp_total_budget.eps, places=3,
                               msg="budget (ctCDP) != budget (DP)")

        # cost (ctCDP) >= budget (ctCDP and DP)
        self.assertGreaterEqual(ctcdp_total_cost.rho, ctcdp_total_budget.rho,
                                msg="cost (ctCDP) < budget (ctCDP)")
        self.assertGreaterEqual(ctcdp_total_cost.to_dp(self.target_delta).eps, dp_total_budget.eps,
                                msg="cost (ctCDP) < budget (DP)")

        # cost (ctCDP) == budget (ctCDP and DP)
        self.assertAlmostEqual(ctcdp_total_cost.rho, ctcdp_total_budget.rho, delta=ctcdp_total_budget.rho*0.1,
                               msg="cost (ctCDP) not equal budget (ctCDP)."
                               )
        self.assertAlmostEqual(ctcdp_total_cost.to_dp(self.target_delta).eps, dp_total_budget.eps, delta=0.01*dp_total_budget.eps,
                               msg="cost (ctCDP) not equal budget (DP).")

    def test_RS_sampling_valid_step(self):
        """Busing uniform schedule, check if the sampling amplification valid, i.e.,
        the cost after amplification is larger than the standard cost (DP) after composition.
        Fail if the metric compose cost is lower than MA or much higher than MA.
        """
        sample_rate = self.sample_rate
        batchs_per_epoch = int(1 / self.sample_rate)
        batch_type = "random"
        T = self.epochs * batchs_per_epoch

        # MA DP from noise
        dp_total_budget = self.compose_by_ma(sample_rate, T)
        # ctCDP: set the global constraint omega first.
        ctcdp_total_budget = privacy_metric.ctCDP.from_dp(dp_total_budget)

        for t in range(1, T, 100):
            # MA
            dp_cost = self.compose_by_ma(sample_rate, t)

            # ctCDP
            ctcdp_cost = self.compose_by_cdp(type(ctcdp_total_budget), sample_rate, t, batch_type)

            # cost (ctCDP) >= cost (DP)
            self.assertGreaterEqual(ctcdp_cost.to_dp(self.target_delta).eps, dp_cost.eps,
                                    msg=f"cost (ctCDP) < cost (DP) at step {t}")

    def test_RF_sampling_valid(self):
        """Uniform schedule.
        We will compare the given metric and MA in composing privacy costs.
        Fail if the metric compose cost is lower than MA or much higher than MA.
        """
        sample_rate = self.sample_rate
        batchs_per_epoch = int(1 / self.sample_rate)
        batch_type = "shuffle"
        T = self.epochs * batchs_per_epoch

        # MA DP from noise
        # dp_total_budget = self.compose_by_ma(sample_rate, T)
        dp_total_budget = self.compose_by_ma(1., self.epochs)

        # ctCDP
        # set the global constraint omega first.
        cdp_total_budget = privacy_metric.zCDP.from_dp(dp_total_budget)
        cdp_total_cost = self.compose_by_cdp(type(cdp_total_budget),
                                                 sample_rate, T, batch_type)

        # budget (ctCDP) == budget (DP)
        self.assertAlmostEqual(cdp_total_budget.to_dp(self.target_delta).eps, dp_total_budget.eps, places=3,
                               msg="budget (ctCDP) != budget (DP)")

        # cost (ctCDP) >= budget (ctCDP and DP)
        self.assertGreaterEqual(cdp_total_cost.rho + 0.0001, cdp_total_budget.rho,
                                msg="cost (ctCDP) < budget (ctCDP)")
        self.assertGreaterEqual(cdp_total_cost.to_dp(self.target_delta).eps+0.001, dp_total_budget.eps,
                                msg="cost (ctCDP) < budget (DP)")

        # cost (ctCDP) == budget (ctCDP and DP)
        self.assertAlmostEqual(cdp_total_cost.rho, cdp_total_budget.rho, delta=cdp_total_budget.rho*0.1,
                               msg="cost (ctCDP) not equal budget (ctCDP)."
                               )
        self.assertAlmostEqual(cdp_total_cost.to_dp(self.target_delta).eps, dp_total_budget.eps, delta=0.01*dp_total_budget.eps,
                               msg="cost (ctCDP) not equal budget (DP).")

    def test_RF_sampling_valid_step(self):
        """Busing uniform schedule, check if the sampling amplification valid, i.e.,
        the cost after amplification is larger than the standard cost (DP) after composition.
        Fail if the metric compose cost is lower than MA or much higher than MA.
        """
        sample_rate = self.sample_rate
        batchs_per_epoch = int(1 / self.sample_rate)
        batch_type = "shuffle"

        # MA DP from noise
        dp_total_budget = self.compose_by_ma(1, self.epochs)
        # ctCDP: set the global constraint omega first.
        ctcdp_total_budget = privacy_metric.zCDP.from_dp(dp_total_budget)

        for ep in range(1, self.epochs, 100):
            t = ep * batchs_per_epoch
            # MA
            dp_cost = self.compose_by_ma(1, ep)

            # ctCDP
            ctcdp_cost = self.compose_by_cdp(type(ctcdp_total_budget), sample_rate, t, batch_type)

            # cost (ctCDP) >= cost (DP)
            self.assertGreaterEqual(ctcdp_cost.to_dp(self.target_delta).eps+0.0001, dp_cost.eps,
                                    msg=f"cost (ctCDP) < cost (DP) at epoch {t}")
