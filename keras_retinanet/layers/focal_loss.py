import keras

"""
As described in https://arxiv.org/abs/1708.02002 .
"""
class FocalLoss(keras.layers.Layer):
	def __init__(self, alpha=0.25, gamma=2.0, *args, **kwargs):
		self.alpha = alpha
		self.gamma = gamma

		super().__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		labels, prediction = inputs

		loss = self.alpha * (1.0 - prediction) ** self.gamma * keras.backend.sparse_categorical_crossentropy(labels, prediction)
		self.add_loss(loss, inputs)
		return loss
