import unittest

import numpy as np
import pandas as pd

from paper_tabpfn.features import design_matrix, context_before
from paper_tabpfn.storage import ub_plan, qb_plan, dp_plan
from track2.config import MIN_ENERGY, MAX_ENERGY
from track2.physics import validate_schedule, settle_day


class StorageStrategyTest(unittest.TestCase):
    def test_flat_prices_do_not_trade(self):
        profit, power = ub_plan(np.full(24, 40.), MIN_ENERGY)
        self.assertEqual(profit, 0)
        self.assertTrue((power == 0).all())

    def test_buys_cheap_hours_and_sells_dear_hours(self):
        prices = np.full(24, 50.)
        prices[10:13] = 0.
        prices[18:20] = 200.
        _, power = ub_plan(prices, MIN_ENERGY)
        self.assertTrue((power[10:13] < 0).all())
        self.assertTrue((power[18:20] > 0).all())
        check = validate_schedule(power, MIN_ENERGY)
        self.assertTrue(check['valid'])
        self.assertAlmostEqual(check['energy'].max(), MAX_ENERGY, places=4)

    def test_random_days_stay_feasible_with_carried_soc(self):
        rng = np.random.default_rng(0)
        energy = MIN_ENERGY
        for _ in range(200):
            forecast = rng.normal(50, 40, 24)
            _, power = ub_plan(forecast, energy)
            result = settle_day(power, forecast + rng.normal(0, 20, 24), energy)
            self.assertTrue(result['valid'], result['reason'])
            energy = result['end_energy']

    def test_dp_feasible_and_at_least_single_loop(self):
        rng = np.random.default_rng(1)
        energy = MIN_ENERGY
        for _ in range(50):
            forecast = rng.normal(50, 40, 48)
            profit, power = dp_plan(forecast, energy)
            self.assertGreaterEqual(profit + 1e-6, ub_plan(forecast[:24], energy)[0] * 0.98)
            result = settle_day(power, forecast[:24], energy)
            self.assertTrue(result['valid'], result['reason'])
            energy = result['end_energy']

    def test_qb_filter_rejects_unprofitable_pessimistic_plan(self):
        prices = np.full(24, 50.)
        prices[18:20] = 80.
        low, high = prices - 30, prices + 30
        self.assertTrue((qb_plan(prices, low, high, MIN_ENERGY) == 0).all())


class FeatureTest(unittest.TestCase):
    def test_context_excludes_target_day_and_price_is_lagged(self):
        dates = pd.date_range('2025-01-01', periods=20).strftime('%Y-%m-%d')
        price = pd.DataFrame(np.arange(20 * 24, dtype=float).reshape(20, 24), index=dates)
        ones = pd.DataFrame(1., index=dates, columns=range(24))
        fuel = pd.DataFrame(dict(gas=1., carbon=1.), index=dates)
        x, y = design_matrix(dict(price=price, load=ones, res=ones, fuel=fuel))
        self.assertEqual(x.loc[dates[10], 'p_lag1_h0'], y.loc[dates[9], 0])
        cx, _ = context_before(x, y, dates[15], 1449)
        self.assertEqual(cx.index.max(), dates[14])


if __name__ == '__main__':
    unittest.main()
