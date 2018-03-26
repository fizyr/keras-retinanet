"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

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

from ..models import retinanet


custom_objects = retinanet.custom_objects


def download_imagenet(backbone):
    if backbone == 'vgg16':
        resource = keras.applications.vgg16.WEIGHTS_PATH_NO_TOP
        checksum = '6d6bbae143d832006294945121d1f1fc'
    elif backbone == 'vgg19':
        resource = keras.applications.vgg19.WEIGHTS_PATH_NO_TOP
        checksum = '253f8cb515780f3b799900260a226db6'
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    weights_path = keras.applications.imagenet_utils.get_file(
        '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(backbone),
        resource,
        cache_subdir='models',
        file_hash=checksum)

    return weights_path


def validate_backbone(backbone):
    allowed_backbones = ['vgg16', 'vgg19']

    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))


def vgg_retinanet(num_classes, backbone='vgg16', inputs=None, modifier=None, **kwargs):
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False)
    elif backbone == 'vgg19':
        vgg = keras.applications.VGG19(input_tensor=inputs, include_top=False)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model
    layer_names = ["block3_pool", "block4_pool", "block5_pool"]
    layer_outputs = [vgg.get_layer(name).output for name in layer_names]
    return retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
