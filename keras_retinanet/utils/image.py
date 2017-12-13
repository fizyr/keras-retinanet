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


def adjust_transform_for_image(image, transform):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    # Move the origin of transformation.
    result = change_transform_origin(transform, colvec(width, height) * 0.5)

    # Scale the translation with the image size.
    result[0:2, 2] *= [width, height]

    return result


def apply_transform(transform, image, annotations, channel_axis, fill_mode, cval):
    # Update transform for image size.
    transform = adjust_transform_for_image(image, transform)

    # Transform the image itself.
    image = keras.preprocessing.image.apply_transform(image, transform, channel_axis=channel_axis, fill_mode=fill_mode, cval=cval)

    # Transform the bounding boxes in the annotations.
    annotations = annotations.copy()
    for index in range(annotations.shape[0]):
        box                    = annotations[index, :4]
        annotations[index, :4] = transform_aabb(transform, box[0], box[1], box[2], box[3])

    return image, annotations


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
