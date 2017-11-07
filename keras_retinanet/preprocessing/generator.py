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

import numpy as np
import random
import threading
import keras

from keras_retinanet.utils.image import preprocess_image, resize_image, random_transform
from keras_retinanet.utils.anchors import anchor_targets


class Generator(object):
    def __init__(
        self,
        image_data_generator,
        batch_size=1,
        group_method='ratio', # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=600,
        image_max_side=1024,
    ):
        self.image_data_generator = image_data_generator
        self.batch_size           = int(batch_size)
        self.group_method         = group_method
        self.shuffle_groups       = shuffle_groups
        self.image_min_side       = image_min_side
        self.image_max_side       = image_max_side

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        raise NotImplementedError("size method not implemented")

    def num_classes(self):
        raise NotImplementedError("num_classes method not implemented")

    def name_to_label(self, name):
        raise NotImplementedError("name_to_label method not implemented")

    def label_to_name(self, label):
        raise NotImplementedError("label_to_name method not implemented")

    def image_aspect_ratio(self, image_index):
        raise NotImplementedError("image_aspect_ratio method not implemented")

    def load_image(self, image_index):
        raise NotImplementedError("load_image method not implemented")

    def load_annotations(self, image_index):
        raise NotImplementedError("load_annotations method not implemented")

    def load_annotations_group(self, group_index):
        return [self.load_annotations(image_index) for image_index in self.groups[group_index]]

    def load_image_group(self, group_index):
        return [self.load_image(image_index) for image_index in self.groups[group_index]]

    def resize_image(self, image):
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess the image (subtract imagenet mean)
            image = preprocess_image(image)

            # randomly transform both image and annotations
            image, annotations = random_transform(image, annotations, self.image_data_generator)

            # resize image
            image, image_scale = self.resize_image(image)

            # apply resizing to annotations too
            annotations[:, :4] *= image_scale

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def group_images(self):
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

        # shuffle groups
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            group_index = self.group_index
            self.group_index = (self.group_index + 1) % len(self.groups)
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at end of epoch
                random.shuffle(self.groups)

        # load images and annotations
        image_group       = self.load_image_group(group_index)
        annotations_group = self.load_annotations_group(group_index)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        # compute labels and regression targets
        labels_group      = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            labels_group[index], regression_group[index] = anchor_targets(max_shape, annotations, self.num_classes(), mask_shape=image.shape)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states           = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch     = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression


        return image_batch, [regression_batch, labels_batch]
