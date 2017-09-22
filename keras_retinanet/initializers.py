import keras

import numpy as np
import math

class PriorProbability(keras.initializers.Initializer):
	"""
	Initializer applies a prior probability.
	"""

	def __init__(self, num_classes=21, probability=0.1):
		self.num_classes = num_classes
		self.probability = probability

	def get_config(self):
		return {
			'num_classes': self.num_classes,
			'probability': self.probability
		}

	def __call__(self, shape, dtype=None):
		assert(shape[0] % self.num_classes == 0)

		# set bias to -log((1 - p)/p) for foregound
		result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

		# set bias to -log(p/(1 - p)) for background
		result[::self.num_classes] = -math.log(self.probability / (1 - self.probability))

		return result
