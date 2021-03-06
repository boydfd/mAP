from unittest import TestCase

from src.box.sliding_window import Box, Point, LabelBox
from src.measure.map import MapCalculator


class TestMapCalculator(TestCase):
    def test_calc_iou_for(self):
        ground_true = Box(Point(297, 260), Point(315, 281))
        detection = Box(Point(292.8720703125, 259.5475769042969),
                        Point(316.13922119140625, 281.2409362792969))
        self.assertAlmostEqual(0.7590295794373748, MapCalculator().calc_iou_for(ground_true, detection),
                               places=10)

    def test_calc_tp_fp(self):
        ground_true = [
            LabelBox(Box(Point(0, 0), Point(1, 1)), 1),
            LabelBox(Box(Point(0, 1), Point(1, 1)), 1),
        ]
        detection = [
            LabelBox(Box(Point(0, 1), Point(1, 1)), 1, 0.8),
            LabelBox(Box(Point(0, 0), Point(1, 1)), 1, 0.9),
            LabelBox(Box(Point(0, 0), Point(1, 1)), 1, 0.4),
        ]
        tp, fp, _ = MapCalculator(0.5).calc_tp_fp(ground_true, detection)
        self.assertEqual([1, 1, 0], tp)
        self.assertEqual([0, 0, 1], fp)


    def test_calc_map_for_one_class(self):
        ground_true = [
            LabelBox(Box(Point(0, 0), Point(1, 1)), 1),
            LabelBox(Box(Point(0, 1), Point(1, 1)), 1),
        ]
        detection = [
            LabelBox(Box(Point(0, 1), Point(1, 1)), 1, 0.8),
            LabelBox(Box(Point(0, 0), Point(1, 1)), 1, 0.6),
            LabelBox(Box(Point(2, 2), Point(3, 3)), 1, 0.7),
        ]
        ap = MapCalculator(0.5).calc_ap_for_one_class(ground_true, detection)
        self.assertAlmostEqual(0.5 * 1 + 0.5 * 2 / 3, ap)
