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

from keras.preprocessing.image import ImageDataGenerator
from keras_retinanet.preprocessing.generator import Generator

import numpy as np
import pytest


class SimpleGenerator(Generator):
    def __init__(self, annotations_group):
        self.annotations_group = annotations_group
        super(SimpleGenerator, self).__init__(ImageDataGenerator(), group_method='none', shuffle_groups=False)

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

    def test_simple_filter(self):
        input_annotations_group = [
            np.array([
                [  0,   0, 10, 10],
                [150, 150, 50, 50]
            ]),
        ]

        expected_annotations_group = [
            np.array([
                [  0,   0, 10, 10],
            ]),
        ]

        simple_generator = SimpleGenerator(input_annotations_group)
        # expect a UserWarning
        with pytest.warns(UserWarning):
            annotations_group = simple_generator.load_annotations_group(simple_generator.groups[0])

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
        ]

        expected_annotations_group = [
            np.array([
                [  0,   0,  10,  10],
                [150, 150, 350, 350],
                [  1,   1,   2,   2]
            ]),
            np.zeros((0, 4)),
            np.array([
                [10, 10, 100, 100]
            ])
        ]

        simple_generator = SimpleGenerator(input_annotations_group)
        # expect a UserWarning
        with pytest.warns(UserWarning):
            annotations_group_0 = simple_generator.load_annotations_group(simple_generator.groups[0])
        with pytest.warns(UserWarning):
            annotations_group_1 = simple_generator.load_annotations_group(simple_generator.groups[1])
        with pytest.warns(UserWarning):
            annotations_group_2 = simple_generator.load_annotations_group(simple_generator.groups[2])

        np.testing.assert_equal([expected_annotations_group[0]], annotations_group_0)
        np.testing.assert_equal([expected_annotations_group[1]], annotations_group_1)
        np.testing.assert_equal([expected_annotations_group[2]], annotations_group_2)
