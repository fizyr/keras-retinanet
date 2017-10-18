import keras

import numpy as np
import math


class PriorProbability(keras.initializers.Initializer):
    """
    Initializer applies a prior probability.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foregound
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result
