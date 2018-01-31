"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, detections, color=(255, 0, 0), generator=None):
    """ Draws detections in an image.

    # Arguments
        image      : The image to draw on.
        detections : A [N, 4 + num_classes] matrix (x1, y1, x2, y2, cls_1, cls_2, ...).
        color      : The color of the boxes.
        generator  : (optional) Generator which can map label to class name.
    """
    draw_boxes(image, detections, color=color)

    # draw labels
    for d in detections:
        label   = np.argmax(d[4:])
        score   = d[4 + label]
        caption = (generator.label_to_name(label) if generator else label) + ': {0:.2f}'.format(score)
        draw_caption(image, d, caption)


def draw_annotations(image, annotations, color=(0, 255, 0), generator=None):
    """ Draws annotations in an image.

    # Arguments
        image       : The image to draw on.
        annotations : A [N, 5] matrix (x1, y1, x2, y2, label).
        color       : The color of the boxes.
        generator   : (optional) Generator which can map label to class name.
    """
    draw_boxes(image, annotations, color)

    # draw labels
    for b in annotations:
        label   = b[4]
        caption = '{}'.format(generator.label_to_name(label) if generator else label)
        draw_caption(image, b, caption)
