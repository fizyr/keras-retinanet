import keras
import keras_retinanet.layers

import numpy as np
import pytest


class TestAnchors(object):
    def test_simple(self):
        # create simple Anchors layer
        anchors_layer = keras_retinanet.layers.Anchors(
            size=32,
            stride=8,
            ratios=np.array([1,], keras.backend.floatx()),
            scales=np.array([1,], keras.backend.floatx()),
        )

        # create fake features input (only shape is used anyway)
        features = np.zeros((1, 2, 2, 1024))
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
    @pytest.mark.xfail
    def test_mini_batch(self):
        # create simple Anchors layer
        anchors_layer = keras_retinanet.layers.Anchors(
            size=32,
            stride=8,
            ratios=np.array([1,], keras.backend.floatx()),
            scales=np.array([1,], keras.backend.floatx()),
        )

        # create fake features input with batch_size=2
        features = np.zeros((2, 2, 2, 1024))
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


class TestTensorReshape(object):
    def test_simple(self):
        # create simple TensorReshape layer
        tensor_reshape_layer = keras_retinanet.layers.TensorReshape(
            target_shape=(-1, 4)
        )

        # create random tensor
        features = np.zeros((1, 2, 2, 1024))
        features = keras.backend.variable(features)

        # compute output
        actual = tensor_reshape_layer.call(features)
        actual = keras.backend.eval(actual)

        # compute expected output
        expected = np.zeros((1, 2 * 2 * 1024 // 4, 4), keras.backend.floatx())

        # assert equality
        np.testing.assert_array_equal(actual, expected)

    def test_mini_batch(self):
        # create simple TensorReshape layer
        tensor_reshape_layer = keras_retinanet.layers.TensorReshape(
            target_shape=(-1, 4)
        )

        # create random tensor
        features = np.zeros((2, 2, 2, 1024))
        features = keras.backend.variable(features)

        # compute output
        actual = tensor_reshape_layer.call(features)
        actual = keras.backend.eval(actual)

        # compute expected output
        expected = np.zeros((2, 2 * 2 * 1024 // 4, 4), keras.backend.floatx())

        # assert equality
        np.testing.assert_array_equal(actual, expected)


class TestNonMaximumSuppression(object):
    def test_simple(self):
        pass

    def test_mini_batch(self):
        pass


class TestUpsampleLike(object):
    def test_simple(self):
        pass

    def test_mini_batch(self):
        pass


class TestRegressBoxe(object):
    def test_simple(self):
        pass

    def test_mini_batch(self):
        pass
