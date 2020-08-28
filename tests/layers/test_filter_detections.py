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

from tensorflow import keras
import keras_retinanet.backend
import keras_retinanet.layers

import numpy as np


class TestFilterDetections(object):
    def test_simple(self):
        # create simple FilterDetections layer
        filter_detections_layer = keras_retinanet.layers.FilterDetections()

        # create simple input
        boxes = np.array([[
            [0, 0, 10, 10],
            [0, 0, 10, 10],  # this will be suppressed
        ]], dtype=keras.backend.floatx())
        boxes = keras.backend.constant(boxes)

        classification = np.array([[
            [0, 0.9],  # this will be suppressed
            [0, 1],
        ]], dtype=keras.backend.floatx())
        classification = keras.backend.constant(classification)

        # compute output
        actual_boxes, actual_scores, actual_labels = filter_detections_layer.call([boxes, classification])
        actual_boxes  = keras.backend.eval(actual_boxes)
        actual_scores = keras.backend.eval(actual_scores)
        actual_labels = keras.backend.eval(actual_labels)

        # define expected output
        expected_boxes = -1 * np.ones((1, 300, 4), dtype=keras.backend.floatx())
        expected_boxes[0, 0, :] = [0, 0, 10, 10]

        expected_scores = -1 * np.ones((1, 300), dtype=keras.backend.floatx())
        expected_scores[0, 0] = 1

        expected_labels = -1 * np.ones((1, 300), dtype=keras.backend.floatx())
        expected_labels[0, 0] = 1

        # assert actual and expected are equal
        np.testing.assert_array_equal(actual_boxes, expected_boxes)
        np.testing.assert_array_equal(actual_scores, expected_scores)
        np.testing.assert_array_equal(actual_labels, expected_labels)

    def test_simple_with_other(self):
        # create simple FilterDetections layer
        filter_detections_layer = keras_retinanet.layers.FilterDetections()

        # create simple input
        boxes = np.array([[
            [0, 0, 10, 10],
            [0, 0, 10, 10],  # this will be suppressed
        ]], dtype=keras.backend.floatx())
        boxes = keras.backend.constant(boxes)

        classification = np.array([[
            [0, 0.9],  # this will be suppressed
            [0, 1],
        ]], dtype=keras.backend.floatx())
        classification = keras.backend.constant(classification)

        other = []
        other.append(np.array([[
            [0, 1234],  # this will be suppressed
            [0, 5678],
        ]], dtype=keras.backend.floatx()))
        other.append(np.array([[
            5678,  # this will be suppressed
            1234,
        ]], dtype=keras.backend.floatx()))
        other = [keras.backend.constant(o) for o in other]

        # compute output
        actual = filter_detections_layer.call([boxes, classification] + other)
        actual_boxes  = keras.backend.eval(actual[0])
        actual_scores = keras.backend.eval(actual[1])
        actual_labels = keras.backend.eval(actual[2])
        actual_other  = [keras.backend.eval(a) for a in actual[3:]]

        # define expected output
        expected_boxes = -1 * np.ones((1, 300, 4), dtype=keras.backend.floatx())
        expected_boxes[0, 0, :] = [0, 0, 10, 10]

        expected_scores = -1 * np.ones((1, 300), dtype=keras.backend.floatx())
        expected_scores[0, 0] = 1

        expected_labels = -1 * np.ones((1, 300), dtype=keras.backend.floatx())
        expected_labels[0, 0] = 1

        expected_other = []
        expected_other.append(-1 * np.ones((1, 300, 2), dtype=keras.backend.floatx()))
        expected_other[-1][0, 0, :] = [0, 5678]
        expected_other.append(-1 * np.ones((1, 300), dtype=keras.backend.floatx()))
        expected_other[-1][0, 0] = 1234

        # assert actual and expected are equal
        np.testing.assert_array_equal(actual_boxes, expected_boxes)
        np.testing.assert_array_equal(actual_scores, expected_scores)
        np.testing.assert_array_equal(actual_labels, expected_labels)

        for a, e in zip(actual_other, expected_other):
            np.testing.assert_array_equal(a, e)

    def test_mini_batch(self):
        # create simple FilterDetections layer
        filter_detections_layer = keras_retinanet.layers.FilterDetections()

        # create input with batch_size=2
        boxes = np.array([
            [
                [0, 0, 10, 10],  # this will be suppressed
                [0, 0, 10, 10],
            ],
            [
                [100, 100, 150, 150],
                [100, 100, 150, 150],  # this will be suppressed
            ],
        ], dtype=keras.backend.floatx())
        boxes = keras.backend.constant(boxes)

        classification = np.array([
            [
                [0, 0.9],  # this will be suppressed
                [0, 1],
            ],
            [
                [1,   0],
                [0.9, 0],  # this will be suppressed
            ],
        ], dtype=keras.backend.floatx())
        classification = keras.backend.constant(classification)

        # compute output
        actual_boxes, actual_scores, actual_labels = filter_detections_layer.call([boxes, classification])
        actual_boxes  = keras.backend.eval(actual_boxes)
        actual_scores = keras.backend.eval(actual_scores)
        actual_labels = keras.backend.eval(actual_labels)

        # define expected output
        expected_boxes = -1 * np.ones((2, 300, 4), dtype=keras.backend.floatx())
        expected_boxes[0, 0, :] = [0, 0, 10, 10]
        expected_boxes[1, 0, :] = [100, 100, 150, 150]

        expected_scores = -1 * np.ones((2, 300), dtype=keras.backend.floatx())
        expected_scores[0, 0] = 1
        expected_scores[1, 0] = 1

        expected_labels = -1 * np.ones((2, 300), dtype=keras.backend.floatx())
        expected_labels[0, 0] = 1
        expected_labels[1, 0] = 0

        # assert actual and expected are equal
        np.testing.assert_array_equal(actual_boxes, expected_boxes)
        np.testing.assert_array_equal(actual_scores, expected_scores)
        np.testing.assert_array_equal(actual_labels, expected_labels)
