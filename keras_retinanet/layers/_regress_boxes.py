import keras
import keras_retinanet

class RegressBoxes(keras.layers.Layer):
	def call(self, inputs, **kwargs):
		anchors, regression = inputs
		return keras_retinanet.backend.bbox_transform_inv(anchors, regression)
