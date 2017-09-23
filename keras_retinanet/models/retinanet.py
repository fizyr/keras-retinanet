import keras
import keras_retinanet


def classification_subnet(num_classes=21, num_anchors=9, feature_size=256, prob_pi=0.1):
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
                **options,
            )
        )

    layers.append(
        keras.layers.Conv2D(
            filters=num_classes * num_anchors,
            kernel_initializer=keras.initializers.zeros(),
            bias_initializer=keras_retinanet.initializers.PriorProbability(num_classes=num_classes, probability=prob_pi),
            name='pyramid_classification',
            **options,
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


def compute_pyramid_features(C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, (1, 1), strides=1, padding='same', name='P5')(C5)
    P5_upsampled = keras_retinanet.layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4')([P5_upsampled, P4])
    P4_upsampled = keras_retinanet.layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, (3, 3), strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3')([P4_upsampled, P3])

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, (3, 3), strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7


def RetinaNet(inputs, backbone, num_classes, training=True, feature_size=256, weights_path=None, nms=True, *args, **kwargs):
    if training:
        image, gt_boxes = inputs
    else:
        image = inputs
    image_shape = keras_retinanet.layers.Dimensions()(image)

    # TODO: Parametrize this
    num_anchors = 9
    _, C3, C4, C5 = backbone.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    pyramid_features = compute_pyramid_features(C3, C4, C5)
    strides          = [8,  16,  32,  64, 128]
    sizes            = [32, 64, 128, 256, 512]

    # construct classification and regression subnets
    classification_layers = classification_subnet(num_classes=num_classes, num_anchors=num_anchors, feature_size=feature_size)
    regression_layers     = regression_subnet(num_anchors=num_anchors, feature_size=feature_size)

    # for all pyramid levels, run classification and regression branch and compute anchors
    classification    = None
    labels            = None
    regression        = None
    regression_target = None
    anchors           = None
    for i, (p, stride, size) in enumerate(zip(pyramid_features, strides, sizes)):
        # run the classification subnet
        cls = p
        for l in classification_layers:
            cls = l(cls)

        # compute labels and bbox_reg_targets
        a       = keras_retinanet.layers.Anchors(stride=stride, anchor_size=size, name='anchors_{}'.format(i))(cls)
        anchors = a if anchors is None else keras.layers.Concatenate(axis=1)([anchors, a])
        if training:
            lb, r  = keras_retinanet.layers.AnchorTarget(name='anchor_target_{}'.format(i))([a, image_shape, gt_boxes])

            labels            = lb if labels is None else keras.layers.Concatenate(axis=1)([labels, lb])
            regression_target = r if regression_target is None else keras.layers.Concatenate(axis=1)([regression_target, r])

        # concatenate classification scores
        cls            = keras_retinanet.layers.TensorReshape((-1, num_classes), name='classification_{}'.format(i))(cls)
        classification = cls if classification is None else keras.layers.Concatenate(axis=1)([classification, cls])

        # run the regression subnet
        reg = p
        for l in regression_layers:
            reg = l(reg)

        reg        = keras_retinanet.layers.TensorReshape((-1, 4), name='boxes_reshaped_{}'.format(i))(reg)
        regression = reg if regression is None else keras.layers.Concatenate(axis=1)([regression, reg])

    # compute classification and regression losses
    classification = keras.layers.Activation('softmax', name='classification_softmax')(classification)
    if training:
        cls_loss, reg_loss = keras_retinanet.layers.FocalLoss(num_classes=num_classes, name='focal_loss')([classification, labels, regression, regression_target])

    # compute resulting boxes
    boxes = keras_retinanet.layers.RegressBoxes(name='boxes')([anchors, regression])
    if nms:
        boxes, classification = keras_retinanet.layers.NonMaximumSuppression(num_classes=num_classes, name='nms')([boxes, classification])

    # construct the model
    if training:
        model = keras.models.Model(inputs=inputs, outputs=[boxes, classification, reg_loss, cls_loss], *args, **kwargs)
    else:
        model = keras.models.Model(inputs=inputs, outputs=[boxes, classification], *args, **kwargs)

    # if set, load pretrained weights
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model
