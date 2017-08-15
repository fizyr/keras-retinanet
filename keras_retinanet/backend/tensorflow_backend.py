import tensorflow
import keras
import keras_retinanet.backend

RPN_FG_FRACTION = 0.5
RPN_BATCHSIZE = 256

def shuffle(x):
	"""
	Modify a sequence by shuffling its contents. This function only shuffles
	the array along the first axis of a multi-dimensional array. The order of
	sub-arrays is changed but their contents remains the same.
	"""
	return tensorflow.random_shuffle(x)

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

	return keras.backend.cast(targets, 'float32')

def overlapping(anchors, gt_boxes, inds_inside):
	"""
	overlaps between the anchors and the gt boxes
	:param anchors: Generated anchors
	:param gt_boxes: Ground truth bounding boxes
	:param inds_inside:
	:return:
	"""

	assert keras.backend.ndim(anchors) == 2
	assert keras.backend.ndim(gt_boxes) == 2
	reference = keras_retinanet.backend.overlap(anchors, gt_boxes)

	gt_argmax_overlaps_inds = keras.backend.argmax(reference, axis=0)

	argmax_overlaps_inds = keras.backend.argmax(reference, axis=1)

	indices = keras.backend.stack([
		tensorflow.range(keras.backend.shape(inds_inside)[0]),
		keras.backend.cast(argmax_overlaps_inds, "int32")
	], axis=0)

	indices = keras.backend.transpose(indices)

	max_overlaps = tensorflow.gather_nd(reference, indices)

	return argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds

def label(y_true, y_pred, inds_inside, RPN_NEGATIVE_OVERLAP=0.3, RPN_POSITIVE_OVERLAP=0.7, clobber_positives=False):
	"""
	Create bbox labels.
	label: 1 is positive, 0 is negative, -1 is do not care

	:param inds_inside: indices of anchors inside image
	:param y_pred: anchors
	:param y_true: ground truth objects

	:return: indices of gt boxes with the greatest overlap, balanced labels
	"""
	ones = keras.backend.ones_like(inds_inside, dtype=keras.backend.floatx())
	labels = ones * -1
	zeros = keras.backend.zeros_like(inds_inside, dtype=keras.backend.floatx())

	argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = overlapping(
		y_pred, y_true, inds_inside)

	# assign bg labels first so that positive labels can clobber them
	if not clobber_positives:
		labels = keras_retinanet.backend.where(
			keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP),
			zeros,
			labels
		)

	# fg label: for each gt, anchor with highest overlap
	indices = keras.backend.expand_dims(gt_argmax_overlaps_inds, axis=1)

	updates = keras.backend.ones_like(gt_argmax_overlaps_inds, dtype=keras.backend.floatx())

	# TODO: generalize unique beyond 1D
	unique_indices, unique_indices_indices = tensorflow.unique(
		keras.backend.reshape(indices, (-1,)), out_idx='int32')
	unique_updates = keras.backend.gather(updates, unique_indices)
	inverse_labels = keras.backend.gather(-1 * labels, unique_indices)
	unique_indices = keras.backend.expand_dims(unique_indices, 1)
	labels = keras_retinanet.backend.scatter_add_tensor(labels, unique_indices, inverse_labels + unique_updates)

	# fg label: above threshold IOU
	labels = keras_retinanet.backend.where(
		keras.backend.greater_equal(max_overlaps, RPN_POSITIVE_OVERLAP),
		ones,
		labels
	)

	if clobber_positives:
		# assign bg labels last so that negative labels can clobber positives
		labels = keras_retinanet.backend.where(
			keras.backend.less(max_overlaps, RPN_NEGATIVE_OVERLAP),
			zeros,
			labels
		)

	return argmax_overlaps_inds, balance(labels)

def balance(labels):
	"""
	balance labels by setting some to -1
	:param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)
	:return: array of labels
	"""

	# subsample positive labels if we have too many
	labels = subsample_positive_labels(labels)

	# subsample negative labels if we have too many
	labels = subsample_negative_labels(labels)

	return labels


def subsample_positive_labels(labels):
	"""
	subsample positive labels if we have too many
	:param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)

	:return:
	"""

	num_fg = int(RPN_FG_FRACTION * RPN_BATCHSIZE)

	fg_inds = keras_retinanet.backend.where(keras.backend.equal(labels, 1))
	num_fg_inds = keras.backend.shape(fg_inds)[0]

	size = num_fg_inds - num_fg

	def more_positive():
		# TODO: try to replace tensorflow
		indices = tensorflow.random_shuffle(
				keras.backend.reshape(fg_inds, (-1,)))[:size]

		updates = tensorflow.ones((size,)) * -1

		inverse_labels = keras.backend.gather(labels, indices) * -1

		indices = keras.backend.reshape(indices, (-1, 1))

		return scatter_add_tensor(labels, indices, inverse_labels + updates)

	def less_positive():
		return labels

	predicate = keras.backend.less_equal(size, 0)

	return tensorflow.cond(predicate, lambda: less_positive(), lambda: more_positive())


def subsample_negative_labels(labels):
	"""
	subsample negative labels if we have too many
	:param labels: array of labels (1 is positive, 0 is negative, -1 is dont care)

	:return:
	"""
	num_bg = RPN_BATCHSIZE - keras.backend.shape(keras_retinanet.backend.where(keras.backend.equal(labels, 1)))[0]
	bg_inds = keras_retinanet.backend.where(keras.backend.equal(labels, 0))
	num_bg_inds = keras.backend.shape(bg_inds)[0]

	size = num_bg_inds - num_bg

	def more_negative():
		indices = keras_retinanet.backend.shuffle(keras.backend.reshape(bg_inds, (-1,)))[:size]

		updates = tensorflow.ones((size,)) * -1

		inverse_labels = keras.backend.gather(labels, indices) * -1

		indices = keras.backend.reshape(indices, (-1, 1))

		return scatter_add_tensor(labels, indices, inverse_labels + updates)

	def less_negative():
		return labels

	predicate = keras.backend.less_equal(size, 0)

	return tensorflow.cond(predicate, lambda: less_negative(), lambda: more_negative())

def scatter_add_tensor(ref, indices, updates, name=None):
	"""
	Adds sparse updates to a variable reference.

	This operation outputs ref after the update is done. This makes it easier to chain operations that need to use the
	reset value.

	Duplicate indices: if multiple indices reference the same location, their contributions add.

	Requires updates.shape = indices.shape + ref.shape[1:].
	:param ref: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16,
		int16, int8, complex64, complex128, qint8, quint8, qint32, half.
	:param indices: A Tensor. Must be one of the following types: int32, int64. A tensor of indices into the first
		dimension of ref.
	:param updates: A Tensor. Must have the same dtype as ref. A tensor of updated values to add to ref
	:param name: A name for the operation (optional).
	:return: Same as ref. Returned as a convenience for operations that want to use the updated values after the update
		is done.
	"""
	with tensorflow.name_scope(name, 'scatter_add_tensor',
			[ref, indices, updates]) as scope:
		ref = tensorflow.convert_to_tensor(ref, name='ref')

		indices = tensorflow.convert_to_tensor(indices, name='indices')

		updates = tensorflow.convert_to_tensor(updates, name='updates')

		ref_shape = tensorflow.shape(ref, out_type=indices.dtype, name='ref_shape')

		scattered_updates = tensorflow.scatter_nd(indices, updates, ref_shape, name='scattered_updates')

		with tensorflow.control_dependencies([tensorflow.assert_equal(
			ref_shape,
			tensorflow.shape(scattered_updates, out_type=indices.dtype))]):
			output = tensorflow.add(ref, scattered_updates, name=scope)

		return output

def unmap(data, count, inds_inside, fill=0):
	""" Unmap a subset of item (data) back to the original set of items (of
	size count) """

	if keras.backend.ndim(data) == 1:
		ret = keras.backend.ones((count,), dtype=keras.backend.floatx()) * fill
		inds_nd = keras.backend.expand_dims(inds_inside)
	else:
		ret = keras.backend.ones((count,) + keras.backend.int_shape(data)[1:],
				dtype=keras.backend.floatx()) * fill
		data = keras.backend.transpose(data)
		data = keras.backend.reshape(data, (-1,))

		inds_ii = keras.backend.tile(inds_inside, [4])
		inds_ii = keras.backend.expand_dims(inds_ii)
		ones = keras.backend.expand_dims(keras.backend.ones_like(inds_inside),
				1)
		inds_coords = keras.backend.concatenate(
				[ones * 0, ones, ones * 2, ones * 3], 0)
		inds_nd = keras.backend.concatenate([inds_ii, inds_coords], 1)
	inverse_ret = tensorflow.squeeze(tensorflow.gather_nd(-1 * ret, inds_nd))
	ret = keras_retinanet.backend.scatter_add_tensor(ret, inds_nd, inverse_ret + data)
	return ret
