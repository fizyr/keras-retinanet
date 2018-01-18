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

import warnings

import keras
import keras_resnet
import keras_resnet.models
from ..models import retinanet

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)

custom_objects = retinanet.custom_objects.copy()
custom_objects.update(keras_resnet.custom_objects)


def download_imagenet(backbone):
    allowed_backbones = [50, 101, 152]
    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    filename = resnet_filename.format(backbone)
    resource = resnet_resource.format(backbone)
    if backbone == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif backbone == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif backbone == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return keras.applications.imagenet_utils.get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )


def resnet_retinanet(num_classes, backbone=50, inputs=None, weights='imagenet', skip_mismatch=True, **kwargs):
    allowed_backbones = [50, 101, 152]
    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # determine which weights to load
    if weights == 'imagenet':
        weights_path = download_imagenet(backbone)
    elif weights is None:
        weights_path = None
    else:
        weights_path = weights

    # create the resnet backbone
    if backbone == 50:
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 101:
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 152:
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)

    # create the full model
    model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone=resnet, **kwargs)

    # optionally load weights
    if weights_path:
        model.load_weights(weights_path, by_name=True, skip_mismatch=skip_mismatch)

    return model


def resnet50_retinanet(num_classes, inputs=None, weights='imagenet', skip_mismatch=True, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone=50, inputs=inputs, weights=weights, skip_mismatch=skip_mismatch, **kwargs)


def resnet101_retinanet(num_classes, inputs=None, weights='imagenet', skip_mismatch=True, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone=101, inputs=inputs, weights=weights, skip_mismatch=skip_mismatch, **kwargs)


def resnet152_retinanet(num_classes, inputs=None, weights='imagenet', skip_mismatch=True, **kwargs):
    return resnet_retinanet(num_classes=num_classes, backbone=152, inputs=inputs, weights=weights, skip_mismatch=skip_mismatch, **kwargs)


def ResNet50RetinaNet(inputs, num_classes, skip_mismatch=True, **kwargs):
    warnings.warn("ResNet50RetinaNet is replaced by resnet50_retinanet and will be removed in a future release.")
    return resnet50_retinanet(num_classes, inputs, *args, skip_mismatch=skip_mismatch, **kwargs)


def ResNet101RetinaNet(inputs, num_classes, skip_mismatch=True, **kwargs):
    warnings.warn("ResNet101RetinaNet is replaced by resnet101_retinanet and will be removed in a future release.")
    return resnet101_retinanet(num_classes, inputs, *args, skip_mismatch=skip_mismatch, **kwargs)


def ResNet152RetinaNet(inputs, num_classes, skip_mismatch=True, **kwargs):
    warnings.warn("ResNet152RetinaNet is replaced by resnet152_retinanet and will be removed in a future release.")
    return resnet152_retinanet(num_classes, inputs, *args, skip_mismatch=skip_mismatch, **kwargs)
