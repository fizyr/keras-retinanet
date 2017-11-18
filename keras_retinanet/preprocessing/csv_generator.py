"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)

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

from keras_retinanet.preprocessing.generator import Generator

import numpy as np
from PIL import Image

import cv2
import csv


class CSVGenerator(Generator):
    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        *args,
        **kwargs
    ):
        self.csv_data_file        = csv_data_file

        self.image_names          = []
        self.image_data           = {}

        # parse the provided class file
        self.classes = {}
        with open(csv_class_file, 'rb') as f_class_in:
            csvreader = csv.reader(f_class_in, delimiter=',')
            for classname, class_id in csvreader:
                self.classes[classname] = int(class_id)

        # csv with img_filepath, x1, y1, x2, y2, class_name
        with open(csv_data_file, 'rb') as f_in:
            csvreader = csv.reader(f_in, delimiter=',')
            for row in csvreader:

                img_filepath, x1, y1, x2, y2, classname = row

                # check if the current class name is correctly present
                if classname not in self.classes:
                    raise ValueError('found class name in data file not present in class file: {}'.format(classname))

                if img_filepath not in self.image_names:
                    self.image_names.append(img_filepath)
                    self.image_data[img_filepath] = []

                self.image_data[img_filepath].append(
                    {'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'class': classname})

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(CSVGenerator, self).__init__(*args, **kwargs)

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        path = self.image_names[image_index]
        # PIL is fast for metadata
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        path = self.image_names[image_index]
        return cv2.imread(path)

    def load_annotations(self, image_index):

        path = self.image_names[image_index]

        annots = self.image_data[path]

        boxes = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):

            class_name = annot['class']

            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])

            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes
