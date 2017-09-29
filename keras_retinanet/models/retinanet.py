import keras
import keras_retinanet

import numpy as np


def classification_subnet(num_classes, num_anchors, feature_size=256, prob_pi=0.1):
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


def run_classification_subnet(num_classes, features, classification_layers, identifier):
    # run the classification subnet
    cls = features
    for l in classification_layers:
        cls = l(cls)

    # reshape output and apply softmax
    cls = keras_retinanet.layers.TensorReshape((-1, num_classes), name='classification_{}'.format(identifier))(cls)
    return keras.layers.Activation('softmax', name='classification_softmax_{}'.format(identifier))(cls)


def regression_subnet(num_classes, num_anchors, feature_size=256):
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


def run_regression_subnet(features, regression_layers, identifier):
    # run the regression subnet
    reg = features
    for l in regression_layers:
        reg = l(reg)

    # reshape output
    return keras_retinanet.layers.TensorReshape((-1, 4), name='boxes_reshaped_{}'.format(identifier))(reg)


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
    feature_size                   = 256,
    weights_path                   = None,
    nms                            = True,
    anchor_sizes                   = None,
    anchor_strides                 = None,
    anchor_ratios                  = None,
    anchor_scales                  = None,
    pyramid_features_func          = pyramid_features,
    classification_subnet_func     = classification_subnet,
    regression_subnet_func         = regression_subnet,
    miscellaneous_subnet_func      = None,
    run_classification_subnet_func = run_classification_subnet,
    run_regression_subnet_func     = run_regression_subnet,
    run_miscellaneous_subnet_func  = None,
    *args, **kwargs
):
    assert((miscellaneous_subnet_func is None and run_miscellaneous_subnet_func is None) or
            (miscellaneous_subnet_func and run_miscellaneous_subnet_func))

    image = inputs

    if anchor_ratios is None:
        anchor_ratios = np.array([0.5, 1, 2], keras.backend.floatx())
    if anchor_scales is None:
        anchor_scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx())
    num_anchors = len(anchor_ratios) * len(anchor_scales)

    _, C3, C4, C5 = backbone.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = pyramid_features_func(C3, C4, C5)
    if anchor_strides is None:
        anchor_strides = [8,  16,  32,  64, 128]
    if anchor_sizes is None:
        anchor_sizes = [32, 64, 128, 256, 512]

    # construct classification and regression subnets
    classification_layers = classification_subnet_func(num_classes=num_classes, num_anchors=num_anchors, feature_size=feature_size)
    regression_layers     = regression_subnet_func(num_classes=num_classes, num_anchors=num_anchors, feature_size=feature_size)

    # construct other miscellaneous layers if possible
    if miscellaneous_subnet_func:
        miscellaneous_layers = miscellaneous_subnet_func(num_classes=num_classes, num_anchors=num_anchors, feature_size=feature_size)
    else:
        miscellaneous_layers = None

    # for all pyramid levels, run classification and regression branch and compute anchors
    classification = None
    regression     = None
    anchors        = None
    miscellaneous  = None
    for i, f in enumerate(features):
        # run classification subnet
        _classification = run_classification_subnet_func(num_classes, f, classification_layers, i)
        classification  = _classification if classification is None else keras.layers.Concatenate(axis=1)([classification, _classification])

        # run regression subnet
        _regression = run_regression_subnet_func(f, regression_layers, i)
        regression  = _regression if regression is None else keras.layers.Concatenate(axis=1)([regression, _regression])

        # compute anchors
        _anchors = keras_retinanet.layers.Anchors(anchor_size=anchor_sizes[i], anchor_stride=anchor_strides[i], anchor_ratios=anchor_ratios, anchor_scales=anchor_scales, name='anchors_{}'.format(i))(f)
        anchors  = _anchors if anchors is None else keras.layers.Concatenate(axis=1)([anchors, _anchors])

        # compute miscellaneous data
        if miscellaneous_layers:
            _miscellaneous = run_miscellaneous_subnet_func(f, miscellaneous_layers, i)
            miscellaneous  = _miscellaneous if miscellaneous is None else keras.layers.Concatenate(axis=1)([miscellaneous, _miscellaneous])

    # concatenate regression, classification and miscellaneous if it exists
    predictions = keras.layers.Concatenate(axis=2, name='predictions')([regression, classification, miscellaneous] if miscellaneous else [regression, classification])

    # apply predicted regression to anchors
    boxes = keras_retinanet.layers.RegressBoxes(name='boxes')([anchors, regression])

    # additionally apply non maximum suppression
    if nms:
        if miscellaneous:
            boxes, classification, miscellaneous = keras_retinanet.layers.NonMaximumSuppression(num_classes=num_classes, name='nms')([boxes, classification, miscellaneous])
        else:
            boxes, classification = keras_retinanet.layers.NonMaximumSuppression(num_classes=num_classes, name='nms')([boxes, classification])

    # concatenate the classification and miscellaneous to the boxes
    detections = keras.layers.Concatenate(axis=2, name='detections')([boxes, classification, miscellaneous] if miscellaneous else [boxes, classification])

    # construct the model
    model = keras.models.Model(inputs=inputs, outputs=[predictions, detections], *args, **kwargs)

    # if set, load pretrained weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model
