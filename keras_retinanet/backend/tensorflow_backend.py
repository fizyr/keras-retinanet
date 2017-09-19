import tensorflow
import keras
import keras_retinanet.backend

def resize_images(*args, **kwargs):
	return tensorflow.image.resize_images(*args, **kwargs)

def zeros(*args, **kwargs):
	# not the same as keras.backend.zeros
	return tensorflow.zeros(*args, **kwargs)

def ones(*args, **kwargs):
	# not the same as keras.backend.ones
	return tensorflow.ones(*args, **kwargs)

def non_max_suppression(*args, **kwargs):
	return tensorflow.image.non_max_suppression(*args, **kwargs)

def scatter_nd(*args, **kwargs):
	return tensorflow.scatter_nd(*args, **kwargs)

def range(*args, **kwargs):
	return tensorflow.range(*args, **kwargs)

def gather_nd(params, indices):
	return tensorflow.gather_nd(params, indices)

def meshgrid(*args, **kwargs):
	return tensorflow.meshgrid(*args, **kwargs)

def where(condition, x=None, y=None):
	return tensorflow.where(condition, x, y)

def bbox_transform(ex_rois, gt_rois):
	"""Compute bounding-box regression targets for an image."""
	ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
	ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
	ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
	ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

	gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
	gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
	gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
	gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

	targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
	targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
	targets_dw = keras.backend.log(gt_widths / ex_widths)
	targets_dh = keras.backend.log(gt_heights / ex_heights)

	targets = keras.backend.stack((targets_dx, targets_dy, targets_dw, targets_dh))

	targets = keras.backend.transpose(targets)

	return keras.backend.cast(targets, keras.backend.floatx())

def overlapping(anchors, gt_boxes):
	"""
	overlaps between the anchors and the gt boxes
	:param anchors: Generated anchors
	:param gt_boxes: Ground truth bounding boxes
	:return:
	"""

	assert keras.backend.ndim(anchors) == 2
	assert keras.backend.ndim(gt_boxes) == 2
	reference = keras_retinanet.backend.overlap(anchors, gt_boxes)

	gt_argmax_overlaps_inds = keras.backend.argmax(reference, axis=0)

	argmax_overlaps_inds = keras.backend.argmax(reference, axis=1)

	indices = keras.backend.stack([
		tensorflow.range(keras.backend.shape(anchors)[0]),
		keras.backend.cast(argmax_overlaps_inds, "int32")
	], axis=0)

	indices = keras.backend.transpose(indices)

	max_overlaps = tensorflow.gather_nd(reference, indices)

	return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds

