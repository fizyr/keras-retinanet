"""
Copyright 2018 vidosits (https://github.com/vidosits/)

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
from keras.applications.densenet import DenseNet, get_file

from ..models import retinanet

origin = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/'
file_name = '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'

custom_objects = retinanet.custom_objects

allowed_backbones = {'densenet121': [6, 12, 24, 16], 'densenet169': [6, 12, 32, 32], 'densenet201': [6, 12, 48, 32]}


def download_imagenet(backbone):
    """ Download pre-trained weights for the specified backbone name. This name is in the format
        {backbone}_weights_tf_dim_ordering_tf_kernels_notop where backbone is the densenet + number of layers (e.g. densenet121).
        For more info check the explanation from the keras densenet script itself:
        https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py
    # Arguments
        backbone    : Backbone name.
    """

    # load weights
    if keras.backend.image_data_format() == 'channels_first':
        raise ValueError('Weights for "channels_first" format are not available.')

    weights_url = origin + file_name.format(backbone)
    weights_path = get_file(file_name.format(backbone), weights_url, cache_subdir='models')

    return weights_path


def validate_backbone(backbone):
    """ Validate the backbone choice.
    # Arguments
        backbone    : Backbone name.
    """

    if backbone not in allowed_backbones:
        raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, list(allowed_backbones.keys())))


def densenet_retinanet(num_classes, backbone='densenet121', inputs=None, modifier=None, **kwargs):
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    blocks = allowed_backbones[backbone]
    densenet = DenseNet(blocks=blocks, input_tensor=inputs, include_top=False, pooling=None, weights=None)

    # get last conv layer from the end of each dense block
    layer_outputs = [densenet.get_layer(name='conv{}_block{}_concat'.format(idx + 2, block_num)).output for idx, block_num in enumerate(blocks)]

    # create the densenet backbone
    densenet = keras.models.Model(inputs=inputs, outputs=layer_outputs[1:], name=densenet.name)

    # invoke modifier if given
    if modifier:
        densenet = modifier(densenet)

    # create the full model
    model = retinanet.retinanet_bbox(inputs=inputs, num_classes=num_classes, backbone_layers=densenet.outputs, **kwargs)

    return model
