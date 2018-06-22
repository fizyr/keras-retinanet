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

from keras_retinanet.preprocessing.generator import Generator

import numpy as np
import pytest


class SimpleGenerator(Generator):
    def __init__(self, annotations_group, num_classes=0, image=None):
        self.annotations_group = annotations_group
        self.num_classes_      = num_classes
        self.image             = image
        super(SimpleGenerator, self).__init__(group_method='none', shuffle_groups=False)

    def num_classes(self):
        return self.num_classes_

    def load_image(self, image_index):
        return self.image

    def size(self):
        return len(self.annotations_group)

    def load_annotations(self, image_index):
        result = self.annotations_group[image_index]
        return result


class TestLoadAnnotationsGroup(object):
    def test_simple(self):
        input_annotations_group = [
            np.array([
                [  0,   0,  10,  10],
                [150, 150, 350, 350]
            ]),
        ]
        expected_annotations_group = input_annotations_group

        simple_generator = SimpleGenerator(input_annotations_group)
        annotations_group = simple_generator.load_annotations_group(simple_generator.groups[0])

        np.testing.assert_equal(expected_annotations_group, annotations_group)

    def test_multiple(self):
        input_annotations_group = [
            np.array([
                [  0,   0,  10,  10],
                [150, 150, 350, 350]
            ]),
            np.array([
                [0, 0, 1, 1]
            ])
        ]
        expected_annotations_group = input_annotations_group

        simple_generator = SimpleGenerator(input_annotations_group)
        annotations_group_0 = simple_generator.load_annotations_group(simple_generator.groups[0])
        annotations_group_1 = simple_generator.load_annotations_group(simple_generator.groups[1])

        np.testing.assert_equal([expected_annotations_group[0]], annotations_group_0)
        np.testing.assert_equal([expected_annotations_group[1]], annotations_group_1)


class TestFilterAnnotations(object):
    def test_simple_filter(self):
        input_annotations_group = [
            np.array([
                [  0,   0, 10, 10],
                [150, 150, 50, 50]
            ]),
        ]

        input_image = np.zeros((500, 500, 3))

        expected_annotations_group = [
            np.array([
                [0, 0, 10, 10],
            ]),
        ]

        simple_generator = SimpleGenerator(input_annotations_group)
        annotations_group = simple_generator.load_annotations_group(simple_generator.groups[0])
        # expect a UserWarning
        with pytest.warns(UserWarning):
            image_group, annotations_group = simple_generator.filter_annotations([input_image], annotations_group, simple_generator.groups[0])

        np.testing.assert_equal(expected_annotations_group, annotations_group)

    def test_multiple_filter(self):
        input_annotations_group = [
            np.array([
                [  0,   0,  10,  10],
                [150, 150,  50,  50],
                [150, 150, 350, 350],
                [350, 350, 150, 150],
                [  1,   1,   2,   2],
                [  2,   2,   1,   1]
            ]),
            np.array([
                [0, 0, -1, -1]
            ]),
            np.array([
                [-10, -10,    0,    0],
                [-10, -10, -100, -100],
                [ 10,  10,  100,  100]
            ]),
            np.array([
                [ 10,  10,  100,  100],
                [ 10,  10,  600,  600]
            ]),
        ]

        input_image = np.zeros((500, 500, 3))

        expected_annotations_group = [
            np.array([
                [  0,   0,  10,  10],
                [150, 150, 350, 350],
                [  1,   1,   2,   2]
            ]),
            np.zeros((0, 4)),
            np.array([
                [10, 10, 100, 100]
            ]),
            np.array([
                [ 10,  10,  100,  100]
            ]),
        ]

        simple_generator = SimpleGenerator(input_annotations_group)
        # expect a UserWarning
        annotations_group_0 = simple_generator.load_annotations_group(simple_generator.groups[0])
        with pytest.warns(UserWarning):
            image_group, annotations_group_0 = simple_generator.filter_annotations([input_image], annotations_group_0, simple_generator.groups[0])

        annotations_group_1 = simple_generator.load_annotations_group(simple_generator.groups[1])
        with pytest.warns(UserWarning):
            image_group, annotations_group_1 = simple_generator.filter_annotations([input_image], annotations_group_1, simple_generator.groups[1])

        annotations_group_2 = simple_generator.load_annotations_group(simple_generator.groups[2])
        with pytest.warns(UserWarning):
            image_group, annotations_group_2 = simple_generator.filter_annotations([input_image], annotations_group_2, simple_generator.groups[2])

        np.testing.assert_equal([expected_annotations_group[0]], annotations_group_0)
        np.testing.assert_equal([expected_annotations_group[1]], annotations_group_1)
        np.testing.assert_equal([expected_annotations_group[2]], annotations_group_2)

    def test_complete(self):
        input_annotations_group = [
            np.array([
                [  0,   0, 50, 50, 0],  # one object of class 0
                [150, 150, 50, 50, 1],  # one object of class 1 with an invalid box
            ], dtype=float)
        ]

        input_image = np.zeros((500, 500, 3), dtype=np.uint8)

        simple_generator = SimpleGenerator(input_annotations_group, image=input_image, num_classes=2)
        # expect a UserWarning
        with pytest.warns(UserWarning):
            _, [_, labels_batch] = simple_generator.next()

        # test that only object with class 0 is present in labels_batch
        labels = np.unique(np.argmax(labels_batch == 1, axis=2))
        assert(len(labels) == 1 and labels[0] == 0), 'Expected only class 0 to be present, but got classes {}'.format(labels)
