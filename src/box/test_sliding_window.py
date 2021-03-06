import math
from unittest import TestCase

from src.box.sliding_window import BoxSize, SlidingWindow, Box, Point, LabelBox, MapSerializerForLabelBox


class TestSlidingWindow(TestCase):
    def test_slide(self):
        images_box_size = BoxSize(2, 2)
        sliding_window_box_size = BoxSize(2, 2)
        sliding_window = SlidingWindow(sliding_window_box_size)
        boxes = sliding_window.slide(images_box_size)
        self.assertEqual([Box(Point(0, 0), Point(2 - 1, 2 - 1))], boxes)

    def test_slide_two(self):
        images_box_size = BoxSize(2 * 2, 2 * 2)
        sliding_window_box_size = BoxSize(2, 2)
        sliding_window = SlidingWindow(sliding_window_box_size)
        boxes = sliding_window.slide(images_box_size)
        expected = [
            Box(Point(0, 0), Point(1, 1)),
            Box(Point(2, 0), Point(3, 1)),
            Box(Point(0, 2), Point(1, 3)),
            Box(Point(2, 2), Point(3, 3)),
        ]
        self.assertEqual(expected, boxes)

    def test_slide_with_insufficient_width_height(self):
        images_box_size = BoxSize(2 * 2, 2 * 2)
        sliding_window_box_size = BoxSize(3, 3)
        sliding_window = SlidingWindow(sliding_window_box_size)
        boxes = sliding_window.slide(images_box_size)
        expected = [
            Box(Point(0, 0), Point(2, 2)),
            Box(Point(1, 0), Point(3, 2)),
            Box(Point(0, 1), Point(2, 3)),
            Box(Point(1, 1), Point(3, 3)),
        ]
        self.assertEqual(expected, boxes)

    def test_slide_with_insufficient_width(self):
        images_box_size = BoxSize(2 * 2, 2 * 2)
        sliding_window_box_size = BoxSize(3, 2)
        sliding_window = SlidingWindow(sliding_window_box_size)
        boxes = sliding_window.slide(images_box_size)
        expected = [
            Box(Point(0, 0), Point(1, 2)),
            Box(Point(2, 0), Point(3, 2)),
            Box(Point(0, 1), Point(1, 3)),
            Box(Point(2, 1), Point(3, 3)),
        ]
        self.assertEqual(expected, boxes)

    def test_window_size_bigger_than_image_size(self):
        images_box_size = BoxSize(4, 4)
        sliding_window_box_size = BoxSize(5, 5)
        sliding_window = SlidingWindow(sliding_window_box_size)
        boxes = sliding_window.slide(images_box_size)
        expected = [
        ]
        self.assertEqual(expected, boxes)

    def test_slide_two_with_overlap_and_insufficient_width_height(self):
        images_box_size = BoxSize(416 * 2, 416 * 2)
        sliding_window_box_size = BoxSize(416, 416)
        sliding_window = SlidingWindow(sliding_window_box_size, 0.2)
        boxes = sliding_window.slide(images_box_size)
        overlapped_size = math.ceil(416 * 0.2)  # 84
        expected = [
            # height0 width 0 1 2
            Box(Point(0, 0), Point(416 - 1, 416 - 1)),
            Box(Point(416 - overlapped_size, 0), Point(416 * 2 - 1 - overlapped_size, 416 - 1)),
            Box(Point(416, 0), Point(416 * 2 - 1, 416 - 1)),

            # height1 width0 1 2
            Box(Point(0, 416 - overlapped_size), Point(416 - 1, 416 * 2 - 1 - overlapped_size)),
            Box(Point(416 - overlapped_size, 416 - overlapped_size),
                Point(416 * 2 - 1 - overlapped_size, 416 * 2 - 1 - overlapped_size)),
            Box(Point(416, 416 - overlapped_size), Point(416 * 2 - 1, 416 * 2 - 1 - overlapped_size)),

            # height1 width0 1 2
            Box(Point(0, 416), Point(416 - 1, 416 * 2 - 1)),
            Box(Point(416 - overlapped_size, 416), Point(416 * 2 - 1 - overlapped_size, 416 * 2 - 1)),
            Box(Point(416, 416), Point(416 * 2 - 1, 416 * 2 - 1)),
        ]
        self.assertEqual(expected, boxes)


class TestBox(TestCase):
    def test_contain_with_1_threshold(self):
        a = Box(Point(0, 0), Point(10, 10))
        b = Box(Point(0, 0), Point(10, 10))
        self.assertTrue(a.contain(b))

    def test_contain_with_0_5_threshold(self):
        a = Box(Point(0, 0), Point(9, 9))
        b = Box(Point(0, 0), Point(9, 19))
        self.assertTrue(a.contain(b, 0.5))
        self.assertFalse(a.contain(b, 0.51))

    def test_contain_any(self):
        a = Box(Point(0, 0), Point(10, 10))
        b = Box(Point(0, 0), Point(10, 20))
        self.assertTrue(a.contain_any([b], 0.5))

    def test_contain_any_false(self):
        a = Box(Point(0, 0), Point(10, 10))
        b = Box(Point(0, 0), Point(10, 20))
        self.assertFalse(a.contain_any([b], 0.53))

    def test_contain_false_when_negative(self):
        a = Box(Point(0, 0), Point(416, 416))
        b = Box(Point(1568, 1014), Point(1738, 1262))
        self.assertFalse(a.contain_any([b]))

    def test_relative_of(self):
        a = Box(Point(3, 3), Point(9, 9))
        b = Box(Point(2, 2), Point(10, 10))
        self.assertEqual(Box(Point(1, 1), Point(7, 7)), a.relative_of(b))

    def test_relative_of_label_box(self):
        a = Box(Point(3, 3), Point(9, 9))
        label_a = LabelBox(a, "a")
        b = Box(Point(2, 2), Point(10, 10))
        self.assertEqual(LabelBox(Box(Point(1, 1), Point(7, 7)), "a"), label_a.relative_of(b))

    def test_serialize_and_deserialize(self):
        box = MapSerializerForLabelBox({1: "啊", 2: "吧"})
        label_box = LabelBox(Box(Point(0, 0), Point(1, 1)), 1, 1)
        label_box2 = LabelBox(Box(Point(0, 0), Point(1, 1)), 1, 1)
        label_box_deserialized = box.deserialize(box.serialize([label_box, label_box2]))
        self.assertEqual([label_box, label_box2], label_box_deserialized)

    def test_deserialize_ground_truth(self):
        box = MapSerializerForLabelBox({1: "啊", 2: "吧"})
        label_box = LabelBox(Box(Point(0, 0), Point(1, 1)), 1, 1)
        label_box_deserialized = box.deserialize("啊 0 0 1 1")
        self.assertEqual([label_box], label_box_deserialized)
