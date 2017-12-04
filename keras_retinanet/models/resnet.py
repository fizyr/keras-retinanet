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
import keras_resnet
import keras_resnet.models
from ..models import retinanet

WEIGHTS_PATH_NO_TOP_50 = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-50-model.keras.h5'
WEIGHTS_PATH_NO_TOP_101 = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-101-model.keras.h5'
WEIGHTS_PATH_NO_TOP_152 = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-152-model.keras.h5'

custom_objects = retinanet.custom_objects.copy()
custom_objects.update(keras_resnet.custom_objects)


def ResNet50RetinaNet(inputs, num_classes, weights='imagenet', *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'ResNet-50-model.keras.h5',
            WEIGHTS_PATH_NO_TOP_50, cache_subdir='models', md5_hash='3e9f4e4f77bbe2c9bec13b53ee1c2319'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet50(image, include_top=False, freeze_bn=True)

    model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone=resnet, *args, **kwargs)
    model.load_weights(weights_path, by_name=True)
    return model


def ResNet101RetinaNet(inputs, num_classes, weights='imagenet', *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'ResNet-101-model.keras.h5',
            WEIGHTS_PATH_NO_TOP_101, cache_subdir='models', md5_hash='05dc86924389e5b401a9ea0348a3213c'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet101(image, include_top=False, freeze_bn=True)

    model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone=resnet, *args, **kwargs)
    model.load_weights(weights_path, by_name=True)
    return model


def ResNet152RetinaNet(inputs, num_classes, weights='imagenet', *args, **kwargs):
    image = inputs

    # load pretrained imagenet weights?
    if weights == 'imagenet':
        weights_path = keras.applications.imagenet_utils.get_file(
            'ResNet-152-model.keras.h5',
            WEIGHTS_PATH_NO_TOP_152, cache_subdir='models', md5_hash='6ee11ef2b135592f8031058820bb9e71'
        )
    else:
        weights_path = weights

    resnet = keras_resnet.models.ResNet152(image, include_top=False, freeze_bn=True)

    model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone=resnet, *args, **kwargs)
    model.load_weights(weights_path, by_name=True)
    return model
