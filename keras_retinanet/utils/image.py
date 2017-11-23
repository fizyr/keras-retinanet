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


def read_image_bgr(path):
    image = np.asarray(PIL.Image.open(path).convert('RGB'))
    return image[:, :, ::-1]


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


def random_transform(
    image,
    boxes,
    image_data_generator,
    seed=None
):
    if seed is None:
        seed = np.random.randint(10000)

    image = image_data_generator.random_transform(image, seed=seed)

    # set fill mode so that masks are not enlarged
    fill_mode = image_data_generator.fill_mode
    image_data_generator.fill_mode = 'constant'

    for index in range(boxes.shape[0]):
        # generate box mask and randomly transform it
        mask = np.zeros_like(image, dtype=np.uint8)
        b = boxes[index, :4].astype(int)
        mask[b[1]:b[3], b[0]:b[2], :] = 255
        mask = image_data_generator.random_transform(mask, seed=seed)[..., 0]
        mask = mask.copy()  # to force contiguous arrays

        # find bounding box again in augmented image
        [i, j] = np.where(mask == 255)
        boxes[index, 0] = float(min(j))
        boxes[index, 1] = float(min(i))
        boxes[index, 2] = float(max(j))
        boxes[index, 3] = float(max(i))

    # restore fill_mode
    image_data_generator.fill_mode = fill_mode

    return image, boxes


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
