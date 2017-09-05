import keras
import keras_retinanet.backend

"""
As described in https://arxiv.org/abs/1708.02002
"""
class FocalLoss(keras.layers.Layer):
	def __init__(self, num_classes=21, alpha=0.75, gamma=2.0, *args, **kwargs):
		self.num_classes = num_classes
		self.alpha = alpha
		self.gamma = gamma

		super(FocalLoss, self).__init__(*args, **kwargs)

	def classification_loss(self, focal_weight, classification, labels):
		cls_loss = focal_weight * keras.backend.sparse_categorical_crossentropy(labels, classification)
		cls_loss = keras.backend.sum(cls_loss)

		# compute the number of anchors assigned to a ground-truth box
		ones           = keras.backend.ones_like(labels)
		zeros          = keras.backend.zeros_like(labels)
		assigned_boxes = keras_retinanet.backend.where(keras.backend.greater(labels, 0), ones, zeros)

		# "The total focal loss of an image is computed as the sum
		# of the focal loss over all ~100k anchors, normalized by the
		# number of anchors assigned to a ground-truth box."
		cls_loss = cls_loss / (keras.backend.maximum(1.0, keras.backend.sum(assigned_boxes) + keras.backend.epsilon()))
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
		classification, labels = inputs

		classification    = keras.backend.reshape(classification, (-1, self.num_classes))
		labels            = keras.backend.reshape(labels, (-1,))
		#regression        = keras.backend.reshape(regression, (-1, 4))
		#regression_target = keras.backend.reshape(labels, (-1, 4))

		indices = keras_retinanet.backend.where(keras.backend.not_equal(labels, -1))

		#regression        = keras_retinanet.backend.gather_nd(regression, indices)
		#regression_target = keras_retinanet.backend.gather_nd(regression_target, indices)
		classification    = keras_retinanet.backend.gather_nd(classification, indices)
		labels            = keras_retinanet.backend.gather_nd(labels, indices)

		# compute alpha as (1 - alpha) for background and alpha for foreground
		foreground_alpha = keras.backend.ones_like(labels) * self.alpha
		background_alpha = 1.0 - foreground_alpha
		alpha            = keras_retinanet.backend.where(keras.backend.equal(labels, 0), background_alpha, foreground_alpha)

		# select classification scores for labeled anchors
		indices          = keras.backend.expand_dims(keras_retinanet.backend.range(keras.backend.shape(labels)[0]), axis=1)
		labeled_indices  = keras.backend.concatenate([indices, keras.backend.expand_dims(keras.backend.cast(labels, 'int32'), axis=1)], axis=1)
		probabilities    = keras_retinanet.backend.gather_nd(classification, labeled_indices)
		focal_weight     = alpha * (1.0 - probabilities) ** self.gamma

		cls_loss = self.classification_loss(focal_weight, classification, labels)
		self.add_loss(cls_loss)

		#reg_loss = self.regression_loss(focal_weight, labels, regression, regression_target)
		#self.add_loss(cls_loss, [labels, regression, regression_target])

		return focal_weight

	#def compute_mask(self, inputs, mask=None):
	#	return [None, None]
