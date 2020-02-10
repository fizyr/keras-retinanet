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
    def __init__(self, bboxes, labels, num_classes=0, image=None):
        assert(len(bboxes) == len(labels))
        self.bboxes       = bboxes
        self.labels       = labels
        self.num_classes_ = num_classes
        self.image        = image
        super(SimpleGenerator, self).__init__(group_method='none', shuffle_groups=False)

    def num_classes(self):
        return self.num_classes_

    def load_image(self, image_index):
        return self.image

    def image_path(self, image_index):
        return ''

    def size(self):
        return len(self.bboxes)

    def load_annotations(self, image_index):
        annotations = {'labels': self.labels[image_index], 'bboxes': self.bboxes[image_index]}
        return annotations


class TestLoadAnnotationsGroup(object):
    def test_simple(self):
        input_bboxes_group = [
            np.array([
                [  0,   0,  10,  10],
                [150, 150, 350, 350]
            ]),
        ]
        input_labels_group = [
            np.array([
                1,
                3
            ]),
        ]
        expected_bboxes_group = input_bboxes_group
        expected_labels_group = input_labels_group

        simple_generator = SimpleGenerator(input_bboxes_group, input_labels_group)
        annotations = simple_generator.load_annotations_group(simple_generator.groups[0])

        assert('bboxes' in annotations[0])
        assert('labels' in annotations[0])
        np.testing.assert_equal(expected_bboxes_group[0], annotations[0]['bboxes'])
        np.testing.assert_equal(expected_labels_group[0], annotations[0]['labels'])

    def test_multiple(self):
        input_bboxes_group = [
            np.array([
                [  0,   0,  10,  10],
                [150, 150, 350, 350]
            ]),
            np.array([
                [0, 0, 50, 50],
            ]),
        ]
        input_labels_group = [
            np.array([
                1,
                0
            ]),
            np.array([
                3
            ])
        ]
        expected_bboxes_group = input_bboxes_group
        expected_labels_group = input_labels_group

        simple_generator = SimpleGenerator(input_bboxes_group, input_labels_group)
        annotations_group_0 = simple_generator.load_annotations_group(simple_generator.groups[0])
        annotations_group_1 = simple_generator.load_annotations_group(simple_generator.groups[1])

        assert('bboxes' in annotations_group_0[0])
        assert('bboxes' in annotations_group_1[0])
        assert('labels' in annotations_group_0[0])
        assert('labels' in annotations_group_1[0])
        np.testing.assert_equal(expected_bboxes_group[0], annotations_group_0[0]['bboxes'])
        np.testing.assert_equal(expected_labels_group[0], annotations_group_0[0]['labels'])
        np.testing.assert_equal(expected_bboxes_group[1], annotations_group_1[0]['bboxes'])
        np.testing.assert_equal(expected_labels_group[1], annotations_group_1[0]['labels'])


class TestFilterAnnotations(object):
    def test_simple_filter(self):
        input_bboxes_group = [
            np.array([
                [  0,   0, 10, 10],
                [150, 150, 50, 50]
            ]),
        ]
        input_labels_group = [
            np.array([
                3,
                1
            ]),
        ]

        input_image = np.zeros((500, 500, 3))

        expected_bboxes_group = [
            np.array([
                [0, 0, 10, 10],
            ]),
        ]
        expected_labels_group = [
            np.array([
                3,
            ]),
        ]

        simple_generator = SimpleGenerator(input_bboxes_group, input_labels_group)
        annotations = simple_generator.load_annotations_group(simple_generator.groups[0])
        # expect a UserWarning
        with pytest.warns(UserWarning):
            image_group, annotations_group = simple_generator.filter_annotations([input_image], annotations, simple_generator.groups[0])

        np.testing.assert_equal(expected_bboxes_group[0], annotations_group[0]['bboxes'])
        np.testing.assert_equal(expected_labels_group[0], annotations_group[0]['labels'])

    def test_multiple_filter(self):
        input_bboxes_group = [
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

        input_labels_group = [
            np.array([
                6,
                5,
                4,
                3,
                2,
                1
            ]),
            np.array([
                0
            ]),
            np.array([
                10,
                11,
                12
            ]),
            np.array([
                105,
                107
            ]),
        ]

        input_image = np.zeros((500, 500, 3))

        expected_bboxes_group = [
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
        expected_labels_group = [
            np.array([
                6,
                4,
                2
            ]),
            np.zeros((0,)),
            np.array([
                12
            ]),
            np.array([
                105
            ]),
        ]

        simple_generator = SimpleGenerator(input_bboxes_group, input_labels_group)
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

        np.testing.assert_equal(expected_bboxes_group[0], annotations_group_0[0]['bboxes'])
        np.testing.assert_equal(expected_labels_group[0], annotations_group_0[0]['labels'])

        np.testing.assert_equal(expected_bboxes_group[1], annotations_group_1[0]['bboxes'])
        np.testing.assert_equal(expected_labels_group[1], annotations_group_1[0]['labels'])

        np.testing.assert_equal(expected_bboxes_group[2], annotations_group_2[0]['bboxes'])
        np.testing.assert_equal(expected_labels_group[2], annotations_group_2[0]['labels'])

    def test_complete(self):
        input_bboxes_group = [
            np.array([
                [  0,   0, 50, 50],
                [150, 150, 50, 50],  # invalid bbox
            ], dtype=float)
        ]

        input_labels_group = [
            np.array([
                5,  # one object of class 5
                3,  # one object of class 3 with an invalid box
            ], dtype=float)
        ]

        input_image = np.zeros((500, 500, 3), dtype=np.uint8)

        simple_generator = SimpleGenerator(input_bboxes_group, input_labels_group, image=input_image, num_classes=6)
        # expect a UserWarning
        with pytest.warns(UserWarning):
            _, [_, labels_batch] = simple_generator[0]

        # test that only object with class 5 is present in labels_batch
        labels = np.unique(np.argmax(labels_batch == 5, axis=2))
        assert(len(labels) == 1 and labels[0] == 0), 'Expected only class 0 to be present, but got classes {}'.format(labels)
