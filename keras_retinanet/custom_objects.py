from . import initializers
from . import layers

custom_objects = {
    'UpsampleLike'          : layers.UpsampleLike,
    'PriorProbability'      : initializers.PriorProbability,
    'RegressBoxes'          : layers.RegressBoxes,
    'NonMaximumSuppression' : layers.NonMaximumSuppression,
    'Anchors'               : layers.Anchors,
}
