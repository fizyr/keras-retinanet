import keras_retinanet.losses
import tensorflow as tf

import numpy as np

import pytest


def test_smooth_l1():
    regression = np.array([
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    ], dtype=tf.keras.backend.floatx())
    regression = tf.keras.backend.variable(regression)

    regression_target = np.array([
        [
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 0.05, 0, 1],
            [0, 0, 1, 0, 0],
        ]
    ], dtype=tf.keras.backend.floatx())
    regression_target = tf.keras.backend.variable(regression_target)

    loss = keras_retinanet.losses.smooth_l1()(regression_target, regression)
    loss = tf.keras.backend.eval(loss)

    assert loss == pytest.approx((((1 - 0.5 / 9) * 2 + (0.5 * 9 * 0.05 ** 2)) / 3))
