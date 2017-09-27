import numpy as np


def anchor_targets(image, gt_boxes, negative_overlap=0.4, positive_overlap=0.5):
    # first create the anchors for this image
    anchors = anchors_for_image(image)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.ones((anchors.shape[0],)) * -1

    # obtain indices of gt boxes with the greatest overlap
    overlaps             = compute_overlap(anchors, gt_boxes[:, :4])
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps         = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign bg labels first so that positive labels can clobber them
    labels[max_overlaps < negative_overlap] = 0

    # fg label: above threshold IOU
    labels[max_overlaps >= positive_overlap] = 1

    # compute box regression targets
    gt_boxes         = gt_boxes[argmax_overlaps_inds]
    bbox_reg_targets = bbox_transform(anchors, gt_boxes)

    # select correct label from gt_boxes
    labels[labels == 1] = gt_boxes[labels == 1, 4]

    return labels, bbox_reg_targets


def anchors_for_image(image, pyramid_levels=5, anchor_ratios=None, anchor_scales=None):
    strides = [2 ** x for x in range(3, 3 + pyramid_levels)]
    sizes   = [2 ** x for x in range(5, 5 + pyramid_levels)]
    shape   = np.array(image.shape[:2])
    for i in range(2):
        shape = (shape + 1) // 2  # skip the first two levels

    all_anchors = np.zeros((0, 4))
    for i in range(pyramid_levels):
        shape           = (shape + 1) // 2
        anchors         = generate_anchors(base_size=sizes[i], ratios=anchor_ratios, scales=anchor_scales)
        shifted_anchors = shift(shape, strides[i], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    base_anchor   = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors       = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                              for i in range(ratio_anchors.shape[0])])

    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((
        x_ctr - 0.5 * (ws - 1),
        y_ctr - 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1),
        y_ctr + 0.5 * (hs - 1)
    ))

    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def bbox_transform(anchors, gt_boxes):
    """Compute bounding-box regression targets for an image."""
    anchor_widths  = anchors[:, 2] - anchors[:, 0] + 1.0
    anchor_heights = anchors[:, 3] - anchors[:, 1] + 1.0
    anchor_ctr_x   = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y   = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x   = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y   = gt_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    targets_dw = np.log(gt_widths / anchor_widths)
    targets_dh = np.log(gt_heights / anchor_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))
    targets = targets.T

    return targets


def compute_overlap(a, b):
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

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0]) + 1
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1]) + 1

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
