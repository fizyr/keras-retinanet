import keras_retinanet.layers
import keras_retinanet.initializers

custom_objects = {
    'UpsampleLike'          : keras_retinanet.layers.UpsampleLike,
    'PriorProbability'      : keras_retinanet.initializers.PriorProbability,
    'AnchorTarget'          : keras_retinanet.layers.AnchorTarget,
    'Dimensions'            : keras_retinanet.layers.Dimensions,
    'RegressBoxes'          : keras_retinanet.layers.RegressBoxes,
    'NonMaximumSuppression' : keras_retinanet.layers.NonMaximumSuppression,
    'FocalLoss'             : keras_retinanet.layers.FocalLoss,
    'TensorReshape'         : keras_retinanet.layers.TensorReshape,
    'Anchors'               : keras_retinanet.layers.Anchors,
}
