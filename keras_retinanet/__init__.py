import keras_retinanet.backend
import keras_retinanet.layers
import keras_retinanet.models
import keras_retinanet.preprocessing
import keras_retinanet.initializers
import keras_retinanet.losses

custom_objects = {
    'UpsampleLike'          : keras_retinanet.layers.UpsampleLike,
    'PriorProbability'      : keras_retinanet.initializers.PriorProbability,
    'RegressBoxes'          : keras_retinanet.layers.RegressBoxes,
    'NonMaximumSuppression' : keras_retinanet.layers.NonMaximumSuppression,
    'TensorReshape'         : keras_retinanet.layers.TensorReshape,
    'Anchors'               : keras_retinanet.layers.Anchors,
}
