from typing import List

from src.box.sliding_window import LabelBox, Box, LabelBoxBatchRepository
from src.measure.measurer import RecallMeasurer, TpMeasurer, \
    FpMeasurer, ApMeasurer, Measurer, F1Measurer, ScoreThresholdMeasurer


class MapCalculator:
    def __init__(self, overlap_threshold: float = 0.5, score_threshold: float = 0.5):
        self.overlap_threshold = overlap_threshold
        self.score_threshold = score_threshold

    @staticmethod
    def build_measurers(tp, fp, score_threshold_index, ground_truth_count):
        return [
            TpMeasurer(tp), FpMeasurer(fp), ScoreThresholdMeasurer(score_threshold_index),
            RecallMeasurer(ground_truth_count), ApMeasurer()
        ]

    def calc_ap_for_one_class(self, ground_truth: List[LabelBox], detection: List[LabelBox]):
        metrics = self.get_ap_metrics_for_one_class(ground_truth, detection)
        return metrics.get_or_create_metric_by_name(ApMeasurer.metric_name)

    def get_ap_metrics_for_one_class(self, ground_truth, detection):
        tp, fp, score_threshold_index = self.calc_tp_fp(ground_truth, detection)
        metrics_calculator = self.build_measurers(tp, fp, score_threshold_index, len(ground_truth))
        metrics = Measurer.measure_all(metrics_calculator)
        return metrics

    def calc_tp_fp(self, ground_truth: List[LabelBox], detection: List[LabelBox]):
        tp = [0] * len(detection)
        fp = tp.copy()
        detection_boxes = detection.copy()
        detection_boxes = sorted(detection_boxes, key=lambda x: x.score, reverse=True)
        ground_truth_unused = [True] * len(ground_truth)
        score_threshold_index = 0
        for detection_index, detection_box in enumerate(detection_boxes):
            overlaps = [self.calc_iou_for(ground_truth_box, detection_box) for ground_truth_box in ground_truth]
            max_overlap = max(overlaps)
            max_index = overlaps.index(max_overlap)
            if ground_truth_unused[max_index] and max_overlap > self.overlap_threshold:
                ground_truth_unused[max_index] = False
                tp[detection_index] = 1
            else:
                fp[detection_index] = 1
            if detection_box.score > self.score_threshold:
                score_threshold_index = detection_index
        return tp, fp, score_threshold_index

    @staticmethod
    def calc_iou_for(box1: Box, box2: Box):
        intersection = box1.intersection(box2)
        box_size = intersection.box_size()
        if box_size.is_positive():
            union = box1.union(box2)
            iou = box_size.area_size() / union.box_size().area_size()
            return iou
        return 0


if __name__ == '__main__':
    ground_truth_repository = LabelBoxBatchRepository("./input/ground-truth/image.txt", {})
    detection_repository = LabelBoxBatchRepository("./input/detection-results/image.txt", {})
    ground_truth = ground_truth_repository.label_to_label_boxes
    detection = detection_repository.label_to_label_boxes
    calculator = MapCalculator(0.5, 0.8)

    sum_ap = 0
    for label, ground_truth_label_boxes in ground_truth.items():
        metrics = calculator.get_ap_metrics_for_one_class(ground_truth_label_boxes, detection.get(label, []))
        ap = metrics.get_or_create_metric_by_name(ApMeasurer.metric_name)
        index = metrics.get_or_create_metric_by_name(ScoreThresholdMeasurer.metric_name)
        recall = metrics.get_or_create_metric_by_name(RecallMeasurer.metric_name)
        f1 = metrics.get_or_create_metric_by_name(F1Measurer.metric_name)
        print("for {label} ap is: {ap}; recall is : {recall}, f1 is {f1}".format(
            label=label,
            ap=ap,
            recall=recall[index] if recall else 0,
            f1=f1[index] if f1 else 0
        ))
        sum_ap += ap

        print("map is {}".format(sum_ap / len(ground_truth.keys())))

