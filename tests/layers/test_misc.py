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


class TestAnchors(object):
    def test_simple(self):
        # create simple Anchors layer
        anchors_layer = keras_retinanet.layers.Anchors(
            size=32,
            stride=8,
            ratios=np.array([1], keras.backend.floatx()),
            scales=np.array([1], keras.backend.floatx()),
        )

        # create fake features input (only shape is used anyway)
        features = np.zeros((1, 2, 2, 1024), dtype=keras.backend.floatx())
        features = keras.backend.variable(features)

        # call the Anchors layer
        anchors = anchors_layer.call(features)
        anchors = keras.backend.eval(anchors)

        # expected anchor values
        expected = np.array([[
            [-12, -12, 20, 20],
            [-4 , -12, 28, 20],
            [-12, -4 , 20, 28],
            [-4 , -4 , 28, 28],
        ]], dtype=keras.backend.floatx())

        # test anchor values
        np.testing.assert_array_equal(anchors, expected)

    # mark test to fail
    def test_mini_batch(self):
        # create simple Anchors layer
        anchors_layer = keras_retinanet.layers.Anchors(
            size=32,
            stride=8,
            ratios=np.array([1], dtype=keras.backend.floatx()),
            scales=np.array([1], dtype=keras.backend.floatx()),
        )

        # create fake features input with batch_size=2
        features = np.zeros((2, 2, 2, 1024), dtype=keras.backend.floatx())
        features = keras.backend.variable(features)

        # call the Anchors layer
        anchors = anchors_layer.call(features)
        anchors = keras.backend.eval(anchors)

        # expected anchor values
        expected = np.array([[
            [-12, -12, 20, 20],
            [-4 , -12, 28, 20],
            [-12, -4 , 20, 28],
            [-4 , -4 , 28, 28],
        ]], dtype=keras.backend.floatx())
        expected = np.tile(expected, (2, 1, 1))

        # test anchor values
        np.testing.assert_array_equal(anchors, expected)


class TestUpsampleLike(object):
    def test_simple(self):
        # create simple UpsampleLike layer
        upsample_like_layer = keras_retinanet.layers.UpsampleLike()

        # create input source
        source   = np.zeros((1, 2, 2, 1), dtype=keras.backend.floatx())
        source   = keras.backend.variable(source)
        target   = np.zeros((1, 5, 5, 1), dtype=keras.backend.floatx())
        expected = target
        target   = keras.backend.variable(target)

        # compute output
        actual = upsample_like_layer.call([source, target])
        actual = keras.backend.eval(actual)

        np.testing.assert_array_equal(actual, expected)

    def test_mini_batch(self):
        # create simple UpsampleLike layer
        upsample_like_layer = keras_retinanet.layers.UpsampleLike()

        # create input source
        source = np.zeros((2, 2, 2, 1), dtype=keras.backend.floatx())
        source = keras.backend.variable(source)

        target   = np.zeros((2, 5, 5, 1), dtype=keras.backend.floatx())
        expected = target
        target   = keras.backend.variable(target)

        # compute output
        actual = upsample_like_layer.call([source, target])
        actual = keras.backend.eval(actual)

        np.testing.assert_array_equal(actual, expected)


class TestRegressBoxes(object):
    def test_simple(self):
        mean = [0, 0, 0, 0]
        std  = [0.2, 0.2, 0.2, 0.2]

        # create simple RegressBoxes layer
        regress_boxes_layer = keras_retinanet.layers.RegressBoxes(mean=mean, std=std)

        # create input
        anchors = np.array([[
            [0 , 0 , 10 , 10 ],
            [50, 50, 100, 100],
            [20, 20, 40 , 40 ],
        ]], dtype=keras.backend.floatx())
        anchors = keras.backend.variable(anchors)

        regression = np.array([[
            [0  , 0  , 0  , 0  ],
            [0.1, 0.1, 0  , 0  ],
            [0  , 0  , 0.1, 0.1],
        ]], dtype=keras.backend.floatx())
        regression = keras.backend.variable(regression)

        # compute output
        actual = regress_boxes_layer.call([anchors, regression])
        actual = keras.backend.eval(actual)

        # compute expected output
        expected = np.array([[
            [0 , 0 , 10  , 10  ],
            [51, 51, 100 , 100 ],
            [20, 20, 40.4, 40.4],
        ]], dtype=keras.backend.floatx())

        np.testing.assert_array_almost_equal(actual, expected, decimal=2)

    # mark test to fail
    def test_mini_batch(self):
        mean = [0, 0, 0, 0]
        std  = [0.2, 0.2, 0.2, 0.2]

        # create simple RegressBoxes layer
        regress_boxes_layer = keras_retinanet.layers.RegressBoxes(mean=mean, std=std)

        # create input
        anchors = np.array([
            [
                [0 , 0 , 10 , 10 ],  # 1
                [50, 50, 100, 100],  # 2
                [20, 20, 40 , 40 ],  # 3
            ],
            [
                [20, 20, 40 , 40 ],  # 3
                [0 , 0 , 10 , 10 ],  # 1
                [50, 50, 100, 100],  # 2
            ],
        ], dtype=keras.backend.floatx())
        anchors = keras.backend.variable(anchors)

        regression = np.array([
            [
                [0  , 0  , 0  , 0  ],  # 1
                [0.1, 0.1, 0  , 0  ],  # 2
                [0  , 0  , 0.1, 0.1],  # 3
            ],
            [
                [0  , 0  , 0.1, 0.1],  # 3
                [0  , 0  , 0  , 0  ],  # 1
                [0.1, 0.1, 0  , 0  ],  # 2
            ],
        ], dtype=keras.backend.floatx())
        regression = keras.backend.variable(regression)

        # compute output
        actual = regress_boxes_layer.call([anchors, regression])
        actual = keras.backend.eval(actual)

        # compute expected output
        expected = np.array([
            [
                [0 , 0 , 10  , 10  ],  # 1
                [51, 51, 100 , 100 ],  # 2
                [20, 20, 40.4, 40.4],  # 3
            ],
            [
                [20, 20, 40.4, 40.4],  # 3
                [0 , 0 , 10  , 10  ],  # 1
                [51, 51, 100 , 100 ],  # 2
            ],
        ], dtype=keras.backend.floatx())

        np.testing.assert_array_almost_equal(actual, expected, decimal=2)
