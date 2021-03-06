from __future__ import annotations

import inspect
import sys
from functools import reduce
from typing import Dict, List, Union

import numpy as np
from graphviz import Digraph


class Metrics:
    def __init__(self, metrics: Dict[str, Union[float, List[float]]]):
        self.metrics = metrics
        self.no_args_measurer: Dict[str, Measurer] = {}

    def add_metric(self, metric_name, metric_value):
        self.metrics[metric_name] = metric_value

    def get_or_create_metric_by_name(self, metric_name: str):
        if metric_name not in self.metrics.keys() and metric_name in self.no_args_measurer.keys():
            self.no_args_measurer[metric_name].measure_(self)
        return self.metrics[metric_name]

    def add_no_args_metric(self, metric_name, param):
        self.no_args_measurer[metric_name] = param

    def copy(self):
        return Metrics(self.metrics)


class Measurer:

    @staticmethod
    def measure_all(measurers: List[Measurer], metrics: Metrics = None) -> Metrics:
        if metrics is None:
            metrics = get_default_metrics()
        metrics = reduce(lambda l, r: r.measure_(l), measurers, metrics)
        return metrics

    def measure(self, metrics: Metrics):
        metrics_copy = metrics.copy()
        return self.measure_(metrics_copy)

    def measure_(self, metrics: Metrics):
        pass

    def return_with_metric_set(self, decrease_precision, metrics: Metrics):
        metrics.add_metric(self.__class__.metric_name, decrease_precision)
        return metrics


class PresetMeasurer(Measurer):
    def __init__(self, preset_metric, metric_name):
        self.preset_metric = preset_metric
        self.metric_name = metric_name

    def measure_(self, metrics: Metrics):
        return self.return_with_metric_set(self.preset_metric, metrics)


class TpMeasurer(PresetMeasurer):
    metric_name = 'tp'

    def __init__(self, preset_metric):
        super().__init__(preset_metric, TpMeasurer.metric_name)

class ScoreThresholdMeasurer(PresetMeasurer):
    metric_name = 'score_threshold_index'

    def __init__(self, preset_metric):
        super().__init__(preset_metric, ScoreThresholdMeasurer.metric_name)

class FpMeasurer(PresetMeasurer):
    metric_name = 'fp'

    def __init__(self, preset_metric):
        super().__init__(preset_metric, FpMeasurer.metric_name)


class AccumulatedMeasurer(Measurer):
    def __init__(self, metric_name, accumulated_metric_name):
        self.metric_name = metric_name
        self.accumulated_metric_name = accumulated_metric_name

    def measure_(self, metrics: Metrics):
        metric_accumulated = metrics.get_or_create_metric_by_name(self.metric_name)
        cumsum = 0
        for i, val in enumerate(metric_accumulated):
            metric_accumulated[i] += cumsum
            cumsum += val
        return self.return_with_metric_set(metric_accumulated, metrics)


class TpAccumulatedMeasurer(AccumulatedMeasurer):
    metric_name = 'tp_accumulated'
    tp_measurer = TpMeasurer

    def __init__(self):
        super().__init__(self.tp_measurer.metric_name, TpAccumulatedMeasurer.metric_name)


class FpAccumulatedMeasurer(AccumulatedMeasurer):
    metric_name = 'fp_accumulated'
    fp_measurer = FpMeasurer

    def __init__(self):
        super().__init__(self.fp_measurer.metric_name, FpAccumulatedMeasurer.metric_name)


class PrecisionMeasurer(Measurer):
    metric_name = 'precision'
    tp_accumulated_measurer = TpAccumulatedMeasurer
    fp_accumulated_measurer = FpAccumulatedMeasurer

    def measure_(self, metrics: Metrics):
        tp_accumulated = metrics.get_or_create_metric_by_name(TpAccumulatedMeasurer.metric_name)
        fp_accumulated = metrics.get_or_create_metric_by_name(FpAccumulatedMeasurer.metric_name)
        precision = [0] * len(tp_accumulated)
        for i in range(0, len(tp_accumulated)):
            precision[i] = float(tp_accumulated[i]) / (fp_accumulated[i] + tp_accumulated[i])
        return self.return_with_metric_set(precision, metrics)


class RecallMeasurer(Measurer):
    metric_name = 'recall'
    tp_accumulated_measurer = TpAccumulatedMeasurer

    def __init__(self, ground_truth_count):
        self.ground_truth_count = ground_truth_count

    def measure_(self, metrics: Metrics):
        tp_accumulated = metrics.get_or_create_metric_by_name(self.tp_accumulated_measurer.metric_name)
        recall = [0] * len(tp_accumulated)
        for i in range(0, len(tp_accumulated)):
            recall[i] = float(tp_accumulated[i]) / self.ground_truth_count

        return self.return_with_metric_set(recall, metrics)


class FullRecallMeasurer(Measurer):
    metric_name = 'full_recall'
    recall_measurer = RecallMeasurer

    def measure_(self, metrics: Metrics):
        recall = metrics.get_or_create_metric_by_name(self.recall_measurer.metric_name)
        full_recall = [0.0] + recall + [1.0]
        return self.return_with_metric_set(full_recall, metrics)


class MonoDecreasingPrecisionMeasurer(Measurer):
    metric_name = 'mono_decreasing_precision'
    precision_measurer = PrecisionMeasurer

    def measure_(self, metrics: Metrics):
        precision = metrics.get_or_create_metric_by_name(self.precision_measurer.metric_name)
        decrease_precision = [0.0] + precision + [0.0]
        for i in range(len(decrease_precision) - 2, -1, -1):
            decrease_precision[i] = max(decrease_precision[i], decrease_precision[i + 1])
        return self.return_with_metric_set(decrease_precision, metrics)


class F1Measurer(Measurer):
    metric_name = 'F1'
    recall_measurer = RecallMeasurer
    precision_measurer = PrecisionMeasurer

    def measure_(self, metrics: Metrics):
        recall = metrics.get_or_create_metric_by_name(self.recall_measurer.metric_name)
        precision = metrics.get_or_create_metric_by_name(self.precision_measurer.metric_name)
        F1 = (np.array(recall) * np.array(precision) / (np.array(precision) + np.array(recall)) * 2).tolist()
        return self.return_with_metric_set(F1, metrics)


class ApMeasurer(Measurer):
    metric_name = 'ap'
    full_recall_measurer = FullRecallMeasurer
    mono_decreasing_precision_measurer = MonoDecreasingPrecisionMeasurer

    def measure_(self, metrics: Metrics):
        full_recall = metrics.get_or_create_metric_by_name(self.full_recall_measurer.metric_name)
        decrease_precision = metrics.get_or_create_metric_by_name(self.mono_decreasing_precision_measurer.metric_name)

        recall_change_point = []
        for i in range(0, len(full_recall) - 1):
            if full_recall[i] != full_recall[i + 1]:
                recall_change_point.append(i + 1)

        ap = 0
        for i in recall_change_point:
            ap += (full_recall[i] - full_recall[i - 1]) * decrease_precision[i]
        return self.return_with_metric_set(ap, metrics)


def get_default_metrics(init_metric=None):
    if init_metric is None:
        init_metric = {}
    metrics = Metrics(init_metric)
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            args = inspect.getfullargspec(obj.__init__).args
            if len(args) == 1 and hasattr(obj, 'metric_name'):
                metrics.add_no_args_metric(obj.metric_name, obj())

    return metrics


class MeasurerRender:
    def __init__(self):
        self.G, self.G_code = self.get_dependencies()

    @classmethod
    def get_dependencies(cls):
        G = Digraph(node_attr={'shape': 'box'},
                    graph_attr={"splines": "ortho"})
        G_code = Digraph(node_attr={'shape': 'box'},
                         graph_attr={"splines": "ortho"},
                         )

        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                depend_clazz = [clazz for name, clazz in inspect.getmembers(obj, lambda a: inspect.isclass(a))
                                if issubclass(clazz, Measurer)]
                G.edges([(str(obj), str(clazz)) for clazz in depend_clazz])
                clazz_ = [(obj.__name__ + ';\\l' + inspect.getsource(obj.measure_)
                           .replace('\n', '\\l')
                           .replace(':', '：')
                           ,
                           clazz.__name__ + ';\\l' + inspect.getsource(clazz.measure_)
                           .replace('\n', '\\l')
                           .replace(':', '：')
                           ) for clazz in
                          depend_clazz]
                print("\n----------------\n".join(["\n".join(a) for a in clazz_]))
                G_code.edges(clazz_)

        return G, G_code

    def render_dependencies(self):
        self.G.render(view=True)

    def render_codes(self):
        self.G_code.render(view=True)


if __name__ == '__main__':

    render = MeasurerRender()
    render.render_codes()
