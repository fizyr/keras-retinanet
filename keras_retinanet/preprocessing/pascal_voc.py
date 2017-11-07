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

import keras_retinanet

import cv2
import os
import numpy as np
from PIL import Image

import cv2
import xml.etree.ElementTree as ET

voc_classes = {
    'aeroplane'   : 0,
    'bicycle'     : 1,
    'bird'        : 2,
    'boat'        : 3,
    'bottle'      : 4,
    'bus'         : 5,
    'car'         : 6,
    'cat'         : 7,
    'chair'       : 8,
    'cow'         : 9,
    'diningtable' : 10,
    'dog'         : 11,
    'horse'       : 12,
    'motorbike'   : 13,
    'person'      : 14,
    'pottedplant' : 15,
    'sheep'       : 16,
    'sofa'        : 17,
    'train'       : 18,
    'tvmonitor'   : 19
}


class PascalVocGenerator(keras_retinanet.preprocessing.Generator):
    def __init__(
        self,
        data_dir,
        set_name,
        *args,
        classes=voc_classes,
        image_extension='.jpg',
        skip_truncated=False,
        skip_difficult=False,
        **kwargs
    ):
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.image_names          = [l.strip() for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(PascalVocGenerator, self).__init__(*args, **kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        return cv2.imread(path)

    def load_annotations(self, image_index):
        boxes = np.zeros((0, 5))

        tree = ET.parse(os.path.join(self.data_dir, 'Annotations', self.image_names[image_index] + '.xml'))
        root = tree.getroot()

        width = float(root.find('size').find('width').text)
        height = float(root.find('size').find('height').text)

        for o in root.iter('object'):
            if int(o.find('truncated').text) and self.skip_truncated:
                continue

            if int(o.find('difficult').text) and self.skip_difficult:
                continue

            box = np.zeros((1, 5))

            class_name = o.find('name').text
            if class_name not in self.classes:
                raise Exception('Class name "{}" not found in classes "{}"'.format(class_name, self.classes))

            box[0, 4] = self.name_to_label(class_name)

            bndbox = o.find('bndbox')
            box[0, 0] = float(bndbox.find('xmin').text) - 1
            box[0, 1] = float(bndbox.find('ymin').text) - 1
            box[0, 2] = float(bndbox.find('xmax').text) - 1
            box[0, 3] = float(bndbox.find('ymax').text) - 1

            boxes = np.append(boxes, box, axis=0)

        return boxes
