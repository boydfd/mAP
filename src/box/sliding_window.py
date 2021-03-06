from __future__ import annotations

import math
from typing import List, Dict, Union

import cv2
import numpy as np
from PIL import ImageDraw

from src.box.repository import Serializer, BatchRepository
from src.logger.logger import debug


class DrawingContext:
    def __init__(self, image_size, colors, font, thickness, class_names):
        self.image_size = image_size
        self.colors = colors
        self.font = font
        self.thickness = thickness
        self.class_names = class_names


class BoxSize:
    def __init__(self, height, width):
        self.width = width
        self.height = height

    @staticmethod
    def from_two_point(left_top: Point, right_bottom: Point) -> BoxSize:
        height = max(0, right_bottom.y - left_top.y + 1)
        width = max(0, right_bottom.x - left_top.x + 1)
        return BoxSize(height, width)

    def __members(self):
        return self.width, self.height

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())

    def non_empty(self):
        return self.width != 0 or self.height != 0

    def is_positive(self):
        return self.width > 0 and self.height > 0

    def area_size(self) -> int:
        return self.width * self.height


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __members(self):
        return self.x, self.y

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())

    def __str__(self):
        return '''{{"x": {}, "y": {}}}'''.format(self.x, self.y)

    def __repr__(self):
        return self.__str__()

    def shift(self, box_size: BoxSize):
        return Point(self.x - box_size.width, self.y - box_size.height)

    def overlapped_left_top(self, other) -> Point:
        return Point(max(self.x, other.x), max(self.y, other.y))

    def overlapped_right_bottom(self, other) -> Point:
        return Point(min(self.x, other.x), min(self.y, other.y))

    def max_left_top(self, other) -> Point:
        return Point(min(self.x, other.x), min(self.y, other.y))

    def max_right_bottom(self, other) -> Point:
        return Point(max(self.x, other.x), max(self.y, other.y))

    def calculate_box_size(self, other) -> BoxSize:
        return BoxSize.from_two_point(self, other)


class Box:
    def __init__(self, left_top: Point, right_bottom: Point):
        self.left_top = left_top
        self.right_bottom = right_bottom

    @staticmethod
    def from_height_width_mode(x, y, width, height) -> Box:
        left_top = Point(int(x), int(y))
        right_bottom = Point(int(x + width), int(y + height))
        return Box(left_top, right_bottom)

    @staticmethod
    def from_height_width_mode_bboxes(bboxes: List) -> Box:
        if len(bboxes) > 1:
            raise Exception("len of bbox > 1")
        return Box.from_height_width_mode(*bboxes[0])

    def extract_from_cv2_img(self, cv2_img):
        return cv2_img[self.left_top.y:self.right_bottom.y + 1, self.left_top.x: self.right_bottom.x + 1]

    def extract_from_pil_img(self, pil_img):
        return pil_img.crop((
            self.left_top.x,
            self.left_top.y,
            self.right_bottom.x + 1,
            self.right_bottom.y + 1,
        ))

    def shift_if_exceed_boundary(self, image_size: BoxSize):
        box_size = BoxSize(max(0, self.right_bottom.y - (image_size.height - 1)),
                           max(0, self.right_bottom.x - (image_size.width - 1)))
        if box_size.non_empty():
            new_box = self.shift(box_size)
            debug("box: {} exceed boundary, shift to: {}".format(self, new_box))
            return new_box
        return self

    def shift(self, box_size: BoxSize):
        new_box = Box(self.left_top.shift(box_size), self.right_bottom.shift(box_size))
        return new_box

    def get_overlapped_box(self, other: Box) -> Box:
        return Box(self.left_top.overlapped_left_top(other.left_top),
                   self.right_bottom.overlapped_right_bottom(other.right_bottom))

    def contain_any(self, boxes: List[Box], threshold=1) -> bool:
        return any(self.contain(box, threshold) for box in boxes)

    def find_contain_items(self, boxes: List[Box], threshold=1) -> List[Box]:
        return [box for box in boxes if self.contain(box, threshold)]

    def contain(self, other: Box, threshold=1) -> bool:
        overlapped_size = self.get_overlapped_box(other).box_size().area_size()
        return float(overlapped_size) / float(other.box_size().area_size()) >= (threshold - 0.000000001)

    def box_size(self) -> BoxSize:
        return self.left_top.calculate_box_size(self.right_bottom)

    def relative_of(self, new_foundation: Box) -> Box:
        box = self.shift(BoxSize(new_foundation.left_top.y, new_foundation.left_top.x))
        return box

    def is_in_boundary(self):
        return self.left_top.x >= 0 and self.left_top.y >= 0

    def __members(self):
        return self.left_top, self.right_bottom

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__members() == other.__members()
        else:
            return False

    def __hash__(self):
        return hash(self.__members())

    def __str__(self):
        return '''{{"left_top": {}, "right_bottom": {}}}'''.format(self.left_top, self.right_bottom)

    def __repr__(self):
        return self.__str__()

    def clone(self):
        return Box(Point(self.left_top.x, self.left_top.y), Point(self.right_bottom.x, self.right_bottom.y))

    def intersection(self, other: Box):
        return self.get_overlapped_box(other)

    def union(self, other: Box):
        return Box(self.left_top.max_left_top(other.left_top),
                   self.right_bottom.max_right_bottom(other.right_bottom))


class SlidingWindow:
    def __init__(self, window_size: BoxSize, overlap_rate: float = 0.0):
        self.window_size = window_size
        self.height_overlap_size = math.ceil(window_size.width * overlap_rate)
        self.width_overlap_size = math.ceil(window_size.height * overlap_rate)

    def slide(self, image_size: BoxSize) -> List[Box]:
        boxes = []
        for y in range(0, image_size.height, self.window_size.height - self.height_overlap_size):
            for x in range(0, image_size.width, self.window_size.width - self.width_overlap_size):
                box = Box(Point(x, y), Point((x + self.window_size.width) - 1, (y + self.window_size.height) - 1))

                fixed_box = box.shift_if_exceed_boundary(image_size)
                if fixed_box.is_in_boundary():
                    boxes.append(fixed_box)
        return boxes


class LabelBox(Box):
    def __init__(self, box: Box, label: Union[str, int], score: float = 1):
        super().__init__(box.left_top, box.right_bottom)
        self.label = label
        self.score = score

    @staticmethod
    def from_height_width_mode_bboxes(bboxes: List, label: str) -> LabelBox:
        if len(bboxes) > 1:
            raise Exception("len of bbox > 1")
        return LabelBox(Box.from_height_width_mode(*bboxes[0]), label)

    def to_five_string(self, labels: List[str]):
        return "{},{},{},{},{}".format(self.left_top.x, self.left_top.y, self.right_bottom.x, self.right_bottom.y,
                                       labels.index(self.label))

    def clone(self):
        box = super().clone()
        self.create_from_box(box)

    def shift(self, box_size: BoxSize) -> LabelBox:
        box = super().shift(box_size)
        return self.create_from_box(box)

    def create_from_box(self, box: Box) -> LabelBox:
        return LabelBox(box, self.label)

    def relative_of(self, new_foundation: Box) -> LabelBox:
        box = super().relative_of(new_foundation)
        return self.create_from_box(box)

    def __members(self):
        return self.left_top, self.right_bottom, self.label, self.score

    def __str__(self):
        return '''{{"left_top": {}, "right_bottom": {}, "label": {}, "score":{} }}'''.format(self.left_top,
                                                                                             self.right_bottom,
                                                                                             self.label, self.score)

    def __eq__(self, other):
        if type(other) is type(self):
            print(self.__members())
            print(other.__members())
            return self.__members() == other.__members()
        else:
            return False

    def draw(self, image):
        draw_box(image, self, '{} {:.2f}'.format(self.label, self.score), False)

    def draw_pil(self, image, context: DrawingContext):
        draw_pil_box(image, self, self.label, self.score, context, False)


def draw_pil_box(image, box: Box, clazz, score, context: DrawingContext, mapping=True):
    font = context.font

    thickness = context.thickness
    top = box.left_top.y - 5
    left = box.left_top.x - 5
    bottom = box.right_bottom.y + 5
    right = box.right_bottom.x + 5
    predicted_class = context.class_names[clazz]
    label = '{} {:.2f}'.format(predicted_class, score)

    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
    right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

    # 画框框
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)
    label = label.encode('utf-8')
    print(label, top, left, bottom, right)

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=context.colors[context.class_names.index(predicted_class)])
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=context.colors[context.class_names.index(predicted_class)])
    draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
    del draw


def draw_box(img, box: Box, label, mapping=True, label_map=None):
    if label_map is None:
        label_map = {}
    font = cv2.FONT_HERSHEY_SIMPLEX

    top = box.left_top.y - 5
    left = box.left_top.x - 5
    bottom = box.right_bottom.y + 5
    right = box.right_bottom.x + 5

    left_top_y = max(0, np.floor(top + 0.5).astype('int32'))
    left_top_x = max(0, np.floor(left + 0.5).astype('int32'))
    right_bottom_y = min(1111111, np.floor(bottom + 0.5).astype('int32'))
    right_bottom_x = min(1111111, np.floor(right + 0.5).astype('int32'))

    bottom_left_corner_of_text = (int(left_top_x), int(left_top_y - 5))
    font_scale = 0.3
    font_color = (255, 255, 255)
    font_thickness = 1

    cv2.putText(img, label_map[label] if mapping else label,
                bottom_left_corner_of_text,
                font,
                font_scale,
                font_color,
                font_thickness)
    line_thickness = 1
    cv2.line(img, (left_top_x, left_top_y),
             (left_top_x, right_bottom_y + 1), (255, 255, 255), thickness=line_thickness)
    cv2.line(img, (left_top_x, left_top_y),
             (right_bottom_x + 1, left_top_y), (255, 255, 255), thickness=line_thickness)
    cv2.line(img, (right_bottom_x + 1, left_top_y),
             (right_bottom_x + 1, right_bottom_y + 1), (255, 255, 255), thickness=line_thickness)
    cv2.line(img, (left_top_x, right_bottom_y + 1),
             (right_bottom_x + 1, right_bottom_y + 1), (255, 255, 255), thickness=line_thickness)


class MapSerializerForLabelBox(Serializer):
    def __init__(self, label_map: Union[Dict[int, str], List[str]]):
        self.label_map = label_map
        self.label_map_inverted = {value: key for key, value in label_map.items()}

    def serialize(self, entities: List[LabelBox]) -> str:
        return "\n".join([self.serialize_one(entity) for entity in entities])

    def serialize_one(self, entity: LabelBox) -> str:
        return "{} {} {} {} {} {}".format(self.label_map[entity.label], entity.score, entity.left_top.x,
                                          entity.left_top.y, entity.right_bottom.x, entity.right_bottom.y)

    def deserialize(self, serialized_entities: str) -> List[LabelBox]:
        return [self.deserialize_one(line) for line in serialized_entities.split("\n")]

    def deserialize_one(self, serialized_entity: str) -> LabelBox:
        values = serialized_entity.split(' ')
        length_without_score = 5
        if len(values) == length_without_score:
            coordinate_index = 1
            score = 1
        else:
            coordinate_index = 2
            score = float(values[1])
        coordinate = values[coordinate_index:]
        return LabelBox(
            Box(Point(float(coordinate[0]), float(coordinate[1])), Point(float(coordinate[2]), float(coordinate[3]))),
            self.label_map_inverted.get(values[0], values[0]), score)


def groupby_unsorted(input, key=lambda x: x):
    yielded = set()
    keys = [key(element) for element in input]
    for i, wantedKey in enumerate(keys):
        if wantedKey not in yielded:
            yield (wantedKey,
                   (input[j] for j in range(i, len(input)) if keys[j] == wantedKey))
        yielded.add(wantedKey)


class LabelBoxBatchRepository(BatchRepository):
    def __init__(self, filename, label_map: Union[Dict[int, str], List[str]]):
        super(LabelBoxBatchRepository, self).__init__(filename, MapSerializerForLabelBox(label_map))
        self.label_to_label_boxes = {key: list(value) for key, value in groupby_unsorted(self.data, lambda x: x.label)}
