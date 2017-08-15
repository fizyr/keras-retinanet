import keras.backend
import keras_retinanet.backend


def inside_image(boxes, im_info, allowed_border=0):
	"""
	Calc indices of boxes which are located completely inside of the image
	whose size is specified by img_info ((height, width, scale)-shaped array).

	:param boxes: (None, 4) tensor containing boxes in original image (x1, y1, x2, y2)
	:param img_info: (height, width, scale)
	:param allowed_border: allow boxes to be outside the image by allowed_border pixels
	:return: (None, 4) indices of boxes completely in original image, (None, 4) tensor of boxes completely inside image
	"""

	indices = keras_retinanet.backend.where(
		(boxes[:, 0] >= -allowed_border) &
		(boxes[:, 1] >= -allowed_border) &
		(boxes[:, 2] < allowed_border + im_info[1]) & # width
		(boxes[:, 3] < allowed_border + im_info[0])   # height
	)

	indices = keras.backend.cast(indices, 'int32')

	gathered = keras.backend.gather(boxes, indices)

	return indices[:, 0], keras.backend.reshape(gathered, [-1, 4])


def shift(shape, stride):
	"""
	Produce shifted anchors based on shape of the map and stride size
	"""
	shift_x = keras.backend.arange(0, shape[0]) * stride
	shift_y = keras.backend.arange(0, shape[1]) * stride

	shift_x, shift_y = keras_retinanet.backend.meshgrid(shift_x, shift_y)
	shift_x = keras.backend.reshape(shift_x, [-1])
	shift_y = keras.backend.reshape(shift_y, [-1])

	shifts = keras.backend.stack([
		shift_x,
		shift_y,
		shift_x,
		shift_y
	], axis=0)

	shifts            = keras.backend.transpose(shifts)
	anchors           = keras_retinanet.backend.anchor()
	number_of_anchors = keras.backend.shape(anchors)[0]

	k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

	shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
	shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

	return shifted_anchors

def overlap(a, b):
	"""
	Parameters
	----------
	a: (N, 4) ndarray of float
	b: (K, 4) ndarray of float
	Returns
	-------
	overlaps: (N, K) ndarray of overlap between boxes and query_boxes
	"""
	area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

	iw = keras.backend.minimum(keras.backend.expand_dims(a[:, 2], 1), b[:, 2]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 0], 1), b[:, 0]) + 1
	ih = keras.backend.minimum(keras.backend.expand_dims(a[:, 3], 1), b[:, 3]) - keras.backend.maximum(keras.backend.expand_dims(a[:, 1], 1), b[:, 1]) + 1

	iw = keras.backend.maximum(iw, 0)
	ih = keras.backend.maximum(ih, 0)

	ua = keras.backend.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), 1) + area - iw * ih

	ua = keras.backend.maximum(ua, 0.0001)

	intersection = iw * ih

	return intersection / ua

def anchor(base_size=16, ratios=None, scales=None):
	"""
	Generates a regular grid of multi-aspect and multi-scale anchor boxes.
	"""
	if ratios is None:
		ratios = keras.backend.cast([0.5, 1, 2], keras.backend.floatx())

	if scales is None:
		scales = keras.backend.cast([8, 16, 32], keras.backend.floatx())
	base_anchor = keras.backend.cast([1, 1, base_size, base_size], keras.backend.floatx()) - 1
	base_anchor = keras.backend.expand_dims(base_anchor, 0)

	ratio_anchors = _ratio_enum(base_anchor, ratios)
	anchors = _scale_enum(ratio_anchors, scales)

	return anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
	"""
	Given a vector of widths (ws) and heights (hs) around a center
	(x_ctr, y_ctr), output a set of anchors (windows).
	"""

	col1 = keras.backend.reshape(x_ctr - 0.5 * (ws - 1), (-1, 1))
	col2 = keras.backend.reshape(y_ctr - 0.5 * (hs - 1), (-1, 1))
	col3 = keras.backend.reshape(x_ctr + 0.5 * (ws - 1), (-1, 1))
	col4 = keras.backend.reshape(y_ctr + 0.5 * (hs - 1), (-1, 1))
	anchors = keras.backend.concatenate((col1, col2, col3, col4), axis=1)

	return anchors


def _ratio_enum(anchor, ratios):
	"""
	Enumerate a set of anchors for each aspect ratio wrt an anchor.
	"""
	w, h, x_ctr, y_ctr = _whctrs(anchor)
	size = w * h
	size_ratios = size / ratios
	ws = keras.backend.round(keras.backend.sqrt(size_ratios))
	hs = keras.backend.round(ws * ratios)
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors


def _scale_enum(anchor, scales):
	"""
	Enumerate a set of anchors for each scale wrt an anchor.
	"""

	w, h, x_ctr, y_ctr = _whctrs(anchor)
	ws = keras.backend.expand_dims(w, 1) * scales
	hs = keras.backend.expand_dims(h, 1) * scales
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors


def _whctrs(anchor):
	"""
	Return width, height, x center, and y center for an anchor (window).
	"""
	w = anchor[:, 2] - anchor[:, 0] + 1
	h = anchor[:, 3] - anchor[:, 1] + 1
	x_ctr = anchor[:, 0] + 0.5 * (w - 1)
	y_ctr = anchor[:, 1] + 0.5 * (h - 1)
	return w, h, x_ctr, y_ctr
