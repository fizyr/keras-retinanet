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

from __future__ import division
import keras
import keras.preprocessing.image
import time
import numpy as np
import cv2
import PIL

from .transform import change_transform_origin, transform_aabb, colvec


def read_image_bgr(path):
    image = np.asarray(PIL.Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def preprocess_image(x):
    # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    # Arguments
        fill_mode:             Same as for keras.preprocessing.image.apply_transform
        cval:                  Same as for keras.preprocessing.image.apply_transform
        data_format:           Same as for keras.preprocessing.image.apply_transform
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        cval                 = 0,
        data_format          = None,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.relative_translation = relative_translation

        if data_format is None:
            data_format = keras.backend.image_data_format()
        self.data_format = data_format

        if data_format == 'channels_first':
            self.channel_axis = 0
        elif data_format == 'channels_last':
            self.channel_axis = 2
        else:
            raise ValueError("invalid data_format, expected 'channels_first' or 'channels_last', got '{}'".format(data_format))


def apply_transform(transform, image, params):
    """ Wrapper around keras.preprocessing.image.apply_transform using TransformParameters. """
    return keras.preprocessing.image.apply_transform(
        image,
        transform,
        channel_axis = params.channel_axis,
        fill_mode    = params.fill_mode,
        cval         = params.cval
    )


def resize_image(img, min_side=600, max_side=1024):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, wich can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
