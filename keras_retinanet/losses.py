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
import keras_retinanet


def focal_loss(alpha=0.25, gamma=2.0):
    def _focal_loss(y_true, y_pred):
        labels         = y_true[0, :, 0]
        classification = y_pred[0, :, :]

        indices        = keras_retinanet.backend.where(keras.backend.not_equal(labels, -1))
        classification = keras_retinanet.backend.gather_nd(classification, indices)
        labels         = keras_retinanet.backend.gather_nd(labels, indices)

        # compute alpha as (1 - alpha) for background and alpha for foreground
        foreground_alpha = keras.backend.ones_like(labels) * alpha
        background_alpha = 1.0 - foreground_alpha
        alpha_factor     = keras_retinanet.backend.where(keras.backend.equal(labels, 0), background_alpha, foreground_alpha)

        # select classification scores for labeled anchors
        indices         = keras.backend.expand_dims(keras_retinanet.backend.range(keras.backend.shape(labels)[0]), axis=1)
        labeled_indices = keras.backend.concatenate([indices, keras.backend.expand_dims(keras.backend.cast(labels, 'int32'), axis=1)], axis=1)
        probabilities   = keras_retinanet.backend.gather_nd(classification, labeled_indices)
        focal_weight    = alpha_factor * (1.0 - probabilities) ** gamma

        cls_loss = focal_weight * keras.backend.sparse_categorical_crossentropy(labels, classification)
        cls_loss = keras.backend.sum(cls_loss)

        # compute the number of anchors assigned to a ground-truth box
        ones           = keras.backend.ones_like(labels)
        zeros          = keras.backend.zeros_like(labels)
        assigned_boxes = keras_retinanet.backend.where(keras.backend.greater(labels, 0), ones, zeros)

        # "The total focal loss of an image is computed as the sum
        # of the focal loss over all ~100k anchors, normalized by the
        # number of anchors assigned to a ground-truth box."
        cls_loss = cls_loss / (keras.backend.maximum(1.0, keras.backend.sum(assigned_boxes)))
        return cls_loss

    return _focal_loss

def regression_loss(y_true, y_pred):
    regression        = y_pred
    regression_target = y_true[:, :, :4]
    labels            = y_true[:, :, 4]

    indices           = keras_retinanet.backend.where(keras.backend.greater(labels, 0))
    regression        = keras_retinanet.backend.gather_nd(regression, indices)
    regression_target = keras_retinanet.backend.gather_nd(regression_target, indices)

    regression_diff = regression - regression_target
    regression_diff = keras.backend.abs(regression_diff)
    regression_diff = keras.backend.sum(regression_diff)
    divisor         = keras.backend.maximum(keras.backend.shape(indices)[0], 1)
    divisor         = keras.backend.cast(divisor, keras.backend.floatx())
    return regression_diff / divisor

    #def _focal_loss(y_true, y_pred):
    #    classification    = y_pred[0, :, 4:]
    #    labels            = y_true[0, :, 4]
    #    regression        = y_pred[0, :, :4]
    #    regression_target = y_true[0, :, :4]

    #    cls_loss = classification_loss(classification, labels)
    #    reg_loss = regression_loss(labels, regression, regression_target)

    #    return cls_loss + reg_loss

    #return _focal_loss
