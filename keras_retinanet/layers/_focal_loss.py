import keras
import keras_retinanet.backend

"""
As described in https://arxiv.org/abs/1708.02002
"""
class FocalLoss(keras.layers.Layer):
	def __init__(self, num_classes=21, alpha=0.25, gamma=2.0, *args, **kwargs):
		self.num_classes = num_classes
		self.alpha = alpha
		self.gamma = gamma

		super().__init__(*args, **kwargs)

	def classification_loss(self, focal_weight, classification, labels):
		cls_loss = focal_weight * keras.backend.sparse_categorical_crossentropy(classification, labels)
		cls_loss = keras.backend.mean(cls_loss)
		return cls_loss

	#def regression_loss(self, focal_weight, labels, regression, regression_target):
	#	regression_diff = regression_target - regression

	#	mask = keras.backend.less_equal(keras.backend.abs(regression_diff), 1.0)

	#	a_x = keras_retinanet.backend.where(keras.backend.not_equal(labels, 0), keras.backend.ones_like(labels), keras.backend.zeros_like(labels))
	#	a_x = keras.backend.cast(a_x, keras.backend.floatx())

	#	a_y = mask * (0.5 * regression_diff * regression_diff) + (1 - mask) * (keras.backend.abs(regression_diff) - 0.5)

	#	a = keras_retinanet.backend.matmul(keras.backend.expand_dims(a_x, 0), a_y)
	#	a = keras.backend.sum(a)

	#	# Divided by anchor overlaps
	#	b = keras.backend.epsilon() + a_x
	#	b = keras.backend.sum(b)

	#	reg_loss = 1.0 * (a / b)
	#	return reg_loss

	def call(self, inputs):
		classification, labels, regression, regression_target = inputs

		classification = keras.backend.reshape(classification, (-1, self.num_classes))
		regression     = keras.backend.reshape(regression, (-1, 4))

		indices = keras_retinanet.backend.where(keras.backend.not_equal(labels, -1))

		regression        = keras_retinanet.backend.gather_nd(regression, indices)
		regression_target = keras_retinanet.backend.gather_nd(regression_target, indices)
		classification    = keras_retinanet.backend.gather_nd(classification, indices)
		labels            = keras_retinanet.backend.gather_nd(labels, indices)

		probabilities = keras.backend.max(classification, axis=1)
		focal_weight = self.alpha * (1.0 - probabilities) ** self.gamma

		cls_loss = self.classification_loss(focal_weight, classification, labels)
		self.add_loss(cls_loss, [labels, classification])

		#reg_loss = self.regression_loss(focal_weight, labels, regression, regression_target)
		#self.add_loss(cls_loss, [labels, regression, regression_target])

		return cls_loss

	#def compute_mask(self, inputs, mask=None):
	#	return [None, None]
