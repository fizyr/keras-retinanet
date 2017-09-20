import keras
import keras_retinanet

class Upsampling(keras.layers.Layer):
	def __init__(self, size, *args, **kwargs):
		self.size = size
		super(Upsampling, self).__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		return keras_retinanet.backend.resize_images(inputs, self.size)
