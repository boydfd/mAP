from unittest import TestCase

import numpy as np

from src.measure.measurer import ApMeasurer, RecallMeasurer, PrecisionMeasurer, Measurer, TpMeasurer, FpMeasurer, \
    get_default_metrics


class TestPrecisionMeasurer(TestCase):
    def test_calc_recall(self):
        tp = [1, 0, 1, 1, 0]
        metrics = {
            TpMeasurer.metric_name: tp
        }
        result = Measurer.measure_all([
            RecallMeasurer(10)
        ], get_default_metrics(metrics))

        np.testing.assert_almost_equal([0.1, 0.1, 0.2, 0.3, 0.3],
                                       result.get_or_create_metric_by_name(RecallMeasurer.metric_name), decimal=3)

    def test_calc_precision(self):
        tp = [1, 0, 1, 1, 0]
        fp = [0, 1, 0, 0, 1]

        metrics = {
            TpMeasurer.metric_name: tp,
            FpMeasurer.metric_name: fp
        }
        result = Measurer.measure_all([
            PrecisionMeasurer()
        ], get_default_metrics(metrics))
        np.testing.assert_almost_equal([1, 0.5, 2.0 / 3.0, 3.0 / 4.0, 3.0 / 5.0],
                                       result.get_or_create_metric_by_name(PrecisionMeasurer.metric_name), decimal=3)

    def test_calc_map_from_precision_and_recall(self):
        rec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        prec = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        metrics = {
            RecallMeasurer.metric_name: rec,
            PrecisionMeasurer.metric_name: prec
        }
        result = Measurer.measure_all([
            ApMeasurer()
        ], get_default_metrics(metrics))
        self.assertAlmostEqual(1.0, result.get_or_create_metric_by_name(ApMeasurer.metric_name))
