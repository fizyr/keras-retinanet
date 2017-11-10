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
import csv


class CSVGenerator(keras_retinanet.preprocessing.Generator):
    def __init__(
        self,
        csv_file,
        set_name,
        *args,
        **kwargs
    ):
        self.csv_file             = csv_file
        self.set_name             = set_name

        self.classes              = {}
        self.image_names          = []
        self.image_data           = {}

        # csv with img_filepath, class, x1, y1, x2, y2, dataset
        with open(csv_file, 'rb') as f_in:
            csvreader = csv.reader(f_in, delimiter=',')
            for row in csvreader:

                img_filepath, x1, y1, x2, y2, classname, dataset = row

                # populate the classes from the CSV automatically and assign them an ID
                if classname not in self.classes:
                    self.classes[classname] = len(self.classes)

                if dataset != set_name:
                    continue

                if img_filepath not in self.image_names:
                    self.image_names.append(img_filepath)
                    self.image_data[img_filepath] = []
                self.image_data[img_filepath].append({'x1':int(x1), 'x2':int(x2), 'y1': int(y1), 'y2':int(y2), 'class':classname})

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(CSVGenerator, self).__init__(*args, **kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        path  = self.image_names[image_index]
        # PIL is fast for metadata
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        path = self.image_names[image_index]
        return cv2.imread(path)

    def load_annotations(self, image_index):
        boxes = np.zeros((0, 5))

        path  = self.image_names[image_index]
        image = Image.open(path)

        width = float(image.width)
        height = float(image.height)

        annots = self.image_data[path]

        for annot in annots:

            box = np.zeros((1, 5))

            class_name = annot['class']

            box[0, 4] = self.name_to_label(class_name)

            bndbox = [annot['x1'], annot['y1'], annot['x2'], annot['y2']]
            box[0, 0] = float(bndbox[0]) - 1
            box[0, 1] = float(bndbox[1]) - 1
            box[0, 2] = float(bndbox[2]) - 1
            box[0, 3] = float(bndbox[3]) - 1

            boxes = np.append(boxes, box, axis=0)

        return boxes
