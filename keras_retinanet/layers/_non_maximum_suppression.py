import keras
import keras_retinanet.backend

class NonMaximumSuppression(keras.layers.Layer):
	def __init__(self, num_classes, nms_threshold=0.4, max_boxes=300, *args, **kwargs):
		self.num_classes   = num_classes
		self.nms_threshold = nms_threshold
		self.max_boxes     = max_boxes
		super(NonMaximumSuppression, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		boxes, classification = inputs

		boxes          = keras.backend.reshape(boxes, (-1, 4))
		classification = keras.backend.reshape(classification, (-1, self.num_classes))

		scores = keras.backend.max(classification, axis=1)
		labels = keras.backend.argmax(classification, axis=1)
		indices = keras_retinanet.backend.where(keras.backend.greater(labels, 0))

		boxes          = keras_retinanet.backend.gather_nd(boxes, indices)
		scores         = keras_retinanet.backend.gather_nd(scores, indices)
		classification = keras_retinanet.backend.gather_nd(classification, indices)

		nms_indices = keras_retinanet.backend.non_max_suppression(boxes, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold)

		boxes          = keras.backend.gather(boxes, nms_indices)
		classification = keras.backend.gather(classification, nms_indices)

		boxes          = keras.backend.expand_dims(boxes, axis=0)
		classification = keras.backend.expand_dims(classification, axis=0)

		return [boxes, classification]

	def compute_output_shape(self, input_shape):
		return [(None, 4), (None, self.num_classes)]

	def compute_mask(self, inputs, mask=None):
		return [None, None]

