import keras.backend
import keras.engine

import keras_retinanet.backend


class AnchorTarget(keras.layers.Layer):
	"""Calculate proposal anchor targets and corresponding labels (label: 1 is positive, 0 is negative, -1 is do not care) for ground truth boxes

	# Arguments
		allowed_border: allow boxes to be outside the image by allowed_border pixels
		clobber_positives: if an anchor statisfied by positive and negative conditions given to negative label
		negative_overlap: IoU threshold below which labels should be given negative label
		positive_overlap: IoU threshold above which labels should be given positive label

	# Input shape
		(# of batches, width of feature map, height of feature map, 2 * # of anchors), (# of samples, 4), (width of feature map, height of feature map, channels)

	# Output shape
		(# of samples, ), (# of samples, 4)
	"""

	def __init__(self, stride, allowed_border=0, clobber_positives=False, negative_overlap=0.3, positive_overlap=0.7, *args, **kwargs):
		self.stride = stride

		self.allowed_border	= allowed_border
		self.clobber_positives = clobber_positives
		self.negative_overlap  = negative_overlap
		self.positive_overlap  = positive_overlap

		super().__init__(*args, **kwargs)

	def call(self, inputs, **kwargs):
		scores, im_info, gt_boxes = inputs

		# TODO: Fix usage of batch index
		shape = im_info[0, :2]

		# TODO: Fix usage of batch index
		gt_boxes = gt_boxes[0]

		print(scores)
		height, width, num_anchors = keras.backend.int_shape(scores)[1:]
		total_anchors = height * width * num_anchors // 2

		# 1. Generate proposals from bbox deltas and shifted anchors
		anchors = keras_retinanet.backend.shift((height, width), self.stride)

		# only keep anchors inside the image
		inds_inside, anchors = keras_retinanet.backend.inside_image(anchors, shape, self.allowed_border)

		# 2. obtain indices of gt boxes with the greatest overlap, balanced labels
		argmax_overlaps_indices, labels = keras_retinanet.backend.label(gt_boxes, anchors, inds_inside, self.negative_overlap, self.positive_overlap, self.clobber_positives)

		gt_boxes = keras.backend.gather(gt_boxes, argmax_overlaps_indices)

		# Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
		bbox_reg_targets = keras_retinanet.backend.bbox_transform(anchors, gt_boxes)

		# TODO: Why is bbox_reg_targets' shape (5, ?, 4)? Why is gt_boxes' shape (None, None, 4) and not (None, 4)?
		bbox_reg_targets = keras.backend.reshape(bbox_reg_targets, (-1, 4))

		# map up to original set of anchors
		labels           = keras_retinanet.backend.unmap(labels, total_anchors, inds_inside, fill=-1)
		bbox_reg_targets = keras_retinanet.backend.unmap(bbox_reg_targets, total_anchors, inds_inside, fill=0)

		# TODO: implement inside and outside weights
		return [labels, bbox_reg_targets]

	def compute_output_shape(self, input_shape):
		return [(None, 1), (None, 4)]

	def compute_mask(self, inputs, mask=None):
		return [None, None]
