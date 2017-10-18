import keras
import keras_retinanet


def focal_loss(alpha=0.25, gamma=2.0):
    def _focal_loss(y_true, y_pred):
        labels         = y_true[0, :, :]
        classification = y_pred[0, :, :]

        anchor_state   = keras.backend.max(labels, axis=1) # -1 for ignore, 0 for background, 1 for object
        indices        = keras_retinanet.backend.where(keras.backend.not_equal(anchor_state, -1))
        classification = keras_retinanet.backend.gather_nd(classification, indices)
        labels         = keras_retinanet.backend.gather_nd(labels, indices)
        anchor_state   = keras_retinanet.backend.gather_nd(anchor_state, indices)

        # select classification scores for labeled anchors
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = keras_retinanet.backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = keras_retinanet.backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)
        cls_loss = keras.backend.sum(cls_loss)

        # "The total focal loss of an image is computed as the sum
        # of the focal loss over all ~100k anchors, normalized by the
        # number of anchors assigned to a ground-truth box."
        cls_loss = cls_loss / (keras.backend.maximum(1.0, keras.backend.sum(anchor_state)))
        return cls_loss

    return _focal_loss

def regression_loss(y_true, y_pred):
    regression        = y_pred[0, :, :]
    regression_target = y_true[0, :, :4]
    labels            = y_true[0, :, 4:]

    anchor_state      = keras.backend.max(labels, axis=1) # -1 for ignore, 0 for background, 1 for object
    indices           = keras_retinanet.backend.where(keras.backend.equal(anchor_state, 1))
    regression        = keras_retinanet.backend.gather_nd(regression, indices)
    regression_target = keras_retinanet.backend.gather_nd(regression_target, indices)

    regression_diff = regression - regression_target
    regression_diff = keras.backend.abs(regression_diff)
    regression_diff = keras.backend.sum(regression_diff)
    divisor         = keras.backend.maximum(keras.backend.shape(indices)[0], 1)
    divisor         = keras.backend.cast(divisor, keras.backend.floatx())
    return regression_diff / divisor
