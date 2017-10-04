import keras
import keras_retinanet

import numpy as np


def classification_subnet(num_classes=91, num_anchors=9, feature_size=256, prob_pi=0.1):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    layers = []
    for i in range(4):
        layers.append(
            keras.layers.Conv2D(
                filters=feature_size,
                activation='relu',
                name='cls_{}'.format(i),
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                **options
            )
        )

    layers.append(
        keras.layers.Conv2D(
            filters=num_classes * num_anchors,
            kernel_initializer=keras.initializers.zeros(),
            bias_initializer=keras_retinanet.initializers.PriorProbability(num_classes=num_classes, probability=prob_pi),
            name='pyramid_classification',
            **options
        )
    )

    return layers


def regression_subnet(num_anchors=9, feature_size=256):
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    layers = []
    for i in range(4):
        layers.append(keras.layers.Conv2D(filters=feature_size, activation='relu', name='reg_{}'.format(i), **options))
    layers.append(keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options))

    return layers


def pyramid_features(C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = keras_retinanet.layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4')([P5_upsampled, P4])
    P4_upsampled = keras_retinanet.layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3')([P4_upsampled, P3])

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7


def RetinaNet(
    inputs,
    backbone,
    num_classes,
    feature_size=256,
    weights_path=None,
    nms=True,
    anchor_sizes=None,
    anchor_strides=None,
    anchor_ratios=None,
    anchor_scales=None,
    *args, **kwargs
):
    image = inputs

    if anchor_ratios is None:
        anchor_ratios = np.array([0.5, 1, 2], keras.backend.floatx())
    if anchor_scales is None:
        anchor_scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())
    num_anchors = len(anchor_ratios) * len(anchor_scales)

    _, C3, C4, C5 = backbone.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = pyramid_features(C3, C4, C5)
    if anchor_strides is None:
        anchor_strides = [8,  16,  32,  64, 128]
    if anchor_sizes is None:
        anchor_sizes = [32, 64, 128, 256, 512]

    # construct classification and regression subnets
    classification_layers = classification_subnet(num_classes=num_classes, num_anchors=num_anchors, feature_size=feature_size)
    regression_layers     = regression_subnet(num_anchors=num_anchors, feature_size=feature_size)

    # for all pyramid levels, run classification and regression branch and compute anchors
    classification = None
    regression     = None
    anchors        = None
    for i, (f, stride, size) in enumerate(zip(features, anchor_strides, anchor_sizes)):
        # run the classification subnet
        cls = f
        for l in classification_layers:
            cls = l(cls)

        # compute labels and bbox_reg_targets
        a       = keras_retinanet.layers.Anchors(anchor_size=size, anchor_stride=stride, anchor_ratios=anchor_ratios, anchor_scales=anchor_scales, name='anchors_{}'.format(i))(cls)
        anchors = a if anchors is None else keras.layers.Concatenate(axis=1)([anchors, a])

        # concatenate classification scores
        cls            = keras_retinanet.layers.TensorReshape((-1, num_classes), name='classification_{}'.format(i))(cls)
        classification = cls if classification is None else keras.layers.Concatenate(axis=1)([classification, cls])

        # run the regression subnet
        reg = f
        for l in regression_layers:
            reg = l(reg)

        reg        = keras_retinanet.layers.TensorReshape((-1, 4), name='boxes_reshaped_{}'.format(i))(reg)
        regression = reg if regression is None else keras.layers.Concatenate(axis=1)([regression, reg])

    # compute classification and regression losses
    classification = keras.layers.Activation('softmax', name='classification_softmax')(classification)

    # concatenate regression and classification
    predictions = keras.layers.Concatenate(axis=2, name='predictions')([regression, classification])

    # apply predicted regression to anchors
    boxes = keras_retinanet.layers.RegressBoxes(name='boxes')([anchors, regression])

    # apply non maximum suppression?
    if nms:
        boxes, classification = keras_retinanet.layers.NonMaximumSuppression(num_classes=num_classes, name='nms')([boxes, classification])

    # concatenate the classification scores to the boxes
    detections = keras.layers.Concatenate(axis=2, name='detections')([boxes, classification])

    # construct the model
    model = keras.models.Model(inputs=inputs, outputs=[predictions, detections], *args, **kwargs)

    # if set, load pretrained weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model
