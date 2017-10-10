import keras
import keras_resnet.models
import keras_retinanet.models

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def ResNet50RetinaNet(inputs, weights='imagenet', *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)

    model = keras_retinanet.models.retinanet_bbox(inputs=inputs, backbone=resnet, *args, **kwargs)
    model.load_weights(weights_path, by_name=True)
    return model
