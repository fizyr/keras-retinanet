"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
import keras.backend as K
from . import backend


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        labels         = y_true
        classification = y_pred

        # compute the divisor: for each image in the batch, we want the number of positive anchors

        # clip the labels to 0, 1 so that we ignore the "ignore" label (-1) in the divisor
        divisor = backend.where(keras.backend.less_equal(labels, 0), keras.backend.zeros_like(labels), labels)
        divisor = keras.backend.max(divisor, axis=2, keepdims=True)
        divisor = keras.backend.cast(divisor, keras.backend.floatx())

        # compute the number of positive anchors
        divisor = keras.backend.sum(divisor, axis=1, keepdims=True)

        #  ensure we do not divide by 0
        divisor = keras.backend.maximum(1.0, divisor)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # normalise by the number of positive anchors for each entry in the minibatch
        cls_loss = cls_loss / divisor

        # filter out "ignore" anchors
        anchor_state = keras.backend.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
        indices      = backend.where(keras.backend.not_equal(anchor_state, -1))

        cls_loss = backend.gather_nd(cls_loss, indices)

        # divide by the size of the minibatch
        return keras.backend.sum(cls_loss) / keras.backend.cast(keras.backend.shape(labels)[0], keras.backend.floatx())

    return _focal


def bbox_overlap_iou(bboxes1, bboxes2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = K.tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = K.tf.split(bboxes2, 4, axis=1)

    xI1 = K.maximum(x11, K.transpose(x21))
    yI1 = K.maximum(y11, K.transpose(y21))

    xI2 = K.minimum(x12, K.transpose(x22))
    yI2 = K.minimum(y12, K.transpose(y22))

    inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    union = (bboxes1_area + K.transpose(bboxes2_area)) - inter_area

    return K.maximum(inter_area / union, 0)


def bbox_iog(predicted, ground_truth):
    x11, y11, x12, y12 = K.tf.split(predicted, 4, axis=1)
    x21, y21, x22, y22 = K.tf.split(ground_truth, 4, axis=1)

    xI1 = K.maximum(x11, K.transpose(x21))
    yI1 = K.maximum(y11, K.transpose(y21))

    xI2 = K.minimum(x12, K.transpose(x22))
    yI2 = K.minimum(y12, K.transpose(y22))

    intersect_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    gt_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    return K.maximum(intersect_area / gt_area, 0)


def smooth_l1_distance(y_true, y_pred, delta=3.):
    sigma_squared = delta ** 2

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = y_pred - y_true

    regression_diff = K.abs(regression_diff)
    regression_loss = backend.where(
        K.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * K.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )
    return regression_loss


def smooth_ln(x, delta):
    cond = K.less_equal(x, delta)
    true_fn = -K.log(1 - x)
    false_fn = ((x - delta) / (1 - delta)) - K.log(1 - delta)
    return backend.where(cond, true_fn, false_fn)


def attraction_term(y_true, y_pred, iou_over_predicted):
    # Найти из y_true бокс с большим IOU для всех y_pred
    # Прогоняем его через smooth_l1
    # Суммируем
    # Делим на количество y_pred
    indices_highest_iou = K.argmax(iou_over_predicted, axis=1)
    gt_highest_iou = K.map_fn(lambda i: K.tf.gather_nd(y_true, [i]), indices_highest_iou, dtype=K.floatx())
    return K.sum(smooth_l1_distance(y_pred, gt_highest_iou)) / K.cast(K.shape(y_pred)[0], K.floatx())


def repulsion_term_gt(y_true, y_pred, iou_over_predicted, alpha):
    # Найти из y_true бокс с вторым по величине IOU
    # Находим IoG между этим боксом и y_true
    # Прогоняем IoG через smooth_ln
    # Суммиируем
    # Делим на количество y_pred

    def two_prediction_exists():
        _, indices_2highest_iou = K.tf.nn.top_k(iou_over_predicted, k=2)
        indices_2highest_iou = indices_2highest_iou[:, 1]
        gt_2highest_iou = K.map_fn(lambda i: K.tf.gather_nd(y_true, [i]), indices_2highest_iou, dtype=K.floatx())
        iog = K.map_fn(lambda x: bbox_iog([x[0]], [x[1]]), (y_pred, gt_2highest_iou), dtype=K.floatx())
        iog = K.squeeze(iog, axis=2)
        return K.sum(smooth_ln(iog, alpha)) / K.cast(K.shape(y_pred)[0], K.floatx())

    def predictions_empty():
        return K.variable(0.0, dtype=K.floatx())

    return K.tf.cond(K.greater(K.shape(iou_over_predicted)[1], 1), two_prediction_exists, predictions_empty)


def repulsion_term_box(y_true, y_pred, betta):
    # Делим все множество y_pred боксов на бокс + цель (Проходимся циклом и оставляем для каждой y_true бокс из y_pred с наибольшим IoU)
    # Находим IoU для каждой пары сочетания (Bi, Bj)
    # Для каждой пары находим отношение smooth_ln(IoU) / IoU + e
    # Суммиируем
    return K.variable(0.0, dtype=K.floatx())


def regression_loss_one(y_true, y_pred):
    # Фильтруем y_pred, оставляя те, у которых IOU > 0,5 хотябы с одним y_true

    image_w, image_h, annotations_len, _ = K.tf.split(y_true[0], 4, axis=0)
    image_w, image_h, annotations_len = image_w[0], image_h[0], K.tf.cast(annotations_len[0], K.tf.int32)
    y_true = y_true[1:annotations_len, :4]
    y_pred = y_pred[:, :4]

    y_true = y_true / K.tf.tile([[image_w, image_h] * 2], K.tf.convert_to_tensor([K.shape(y_true)[0], 1]))
    y_pred = y_pred / K.tf.tile([[image_w, image_h] * 2], K.tf.convert_to_tensor([K.shape(y_pred)[0], 1]))

    iou_over_predicted = bbox_overlap_iou(y_pred, y_true)
    highest_iou = K.max(iou_over_predicted, axis=1)

    iou_gt_05 = backend.where(K.greater(highest_iou, 0.5))
    y_pred = K.tf.gather_nd(y_pred, iou_gt_05)
    iou_over_predicted = K.tf.gather_nd(iou_over_predicted, iou_gt_05)

    alpha = 0.5
    beta = 0.5
    has_data = K.tf.logical_and(K.greater(K.shape(iou_over_predicted)[0], 0),
                                K.greater(K.shape(iou_over_predicted)[1], 0))
    has_data = K.tf.logical_and(has_data, K.greater(K.shape(y_true)[0], 0))

    return K.tf.cond(has_data,
                     lambda: K.sum([
                         attraction_term(y_true, y_pred, iou_over_predicted),
                         repulsion_term_gt(y_true, y_pred, iou_over_predicted, alpha),
                         # repulsion_term_box(y_true, y_pred_masked, beta)
                     ]),
                     lambda: K.variable(0.0, dtype=K.floatx()))


def repulsion_loss(y_true, y_pred):
    with K.tf.device('/cpu:0'):
        return K.map_fn(lambda x: regression_loss_one(x[0], x[1]), (y_true, y_pred), dtype=K.floatx())


# def repulsion_loss(y_true, y_pred):
#     return K.variable(0.0, dtype=K.floatx())


def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, 4]

        # compute the divisor: for each image in the batch, we want the number of positive anchors
        divisor = backend.where(K.equal(anchor_state, 1), K.ones_like(anchor_state), K.zeros_like(anchor_state))
        divisor = K.sum(divisor, axis=1, keepdims=True)
        divisor = K.maximum(1.0, divisor)

        # pad the tensor to have shape (batch_size, 1, 1) for future division
        divisor   = K.expand_dims(divisor, axis=2)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target

        regression_diff = K.abs(regression_diff)
        regression_loss = backend.where(
            K.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * K.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # normalise by the number of positive and negative anchors for each entry in the minibatch
        regression_loss = regression_loss / divisor

        # filter out "ignore" anchors
        indices = backend.where(K.equal(anchor_state, 1))
        regression_loss = backend.gather_nd(regression_loss, indices)

        # divide by the size of the minibatch
        regression_loss = K.sum(regression_loss) / K.cast(K.shape(y_true)[0], K.floatx())

        return regression_loss

    return _smooth_l1
