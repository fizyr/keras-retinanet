#!/usr/bin/env python

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

import argparse
import os
import sys
import cv2
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

import keras

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..utils.transform import random_transform_generator
from ..utils.visualization import draw_annotations, draw_boxes
from ..utils.anchors import anchors_for_shape
from .. import backend


def create_generator(args):
    # create random transform generator for augmenting training data
    transform_generator = None
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        generator = CocoGenerator(
            args.coco_path,
            args.coco_set,
            transform_generator=transform_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'pascal':
        generator = PascalVocGenerator(
            args.pascal_path,
            args.pascal_set,
            transform_generator=transform_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'csv':
        generator = CSVGenerator(
            args.annotations,
            args.classes,
            transform_generator=transform_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'oid':
        generator = OpenImagesGenerator(
            args.main_dir,
            subset=args.subset,
            version=args.version,
            labels_filter=args.labels_filter,
            fixed_labels=args.fixed_labels,
            annotation_cache_dir=args.annotation_cache_dir,
            transform_generator=transform_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'kitti':
        generator = KittiGenerator(
            args.kitti_path,
            subset=args.subset,
            transform_generator=transform_generator
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path',  help='Path to dataset directory (ie. /tmp/COCO).')
    coco_parser.add_argument('--coco-set', help='Name of the set to show (defaults to val2017).', default='val2017')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--pascal-set',  help='Name of the set to show (defaults to test).', default='test')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')
    kitti_parser.add_argument('subset', help='Argument for loading a subset from train/val.')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('subset', help='Argument for loading a subset from train/validation/test.')
    oid_parser.add_argument('--version',  help='The current dataset version is V3.', default='2017_11')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--fixed-labels', help='Use the exact specified labels.', default=False)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('-l', '--loop', help='Loop forever, even if the dataset is exhausted.', action='store_true')
    parser.add_argument('--batch-size', help='Batch size of the generator.', type=int, default=1)
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')

    return parser.parse_args(args)


def run(generator, args):
    # display images, one at a time
    for group in generator.groups:
        inputs, targets, annotations = generator.compute_input_output(group)

        # compute regressed anchors
        anchor_state = targets[0][:, :, 4]
        anchors      = anchors_for_shape(inputs.shape[1:])
        anchors      = np.tile(anchors, (args.batch_size, 1, 1))
        anchors_     = keras.backend.variable(anchors)
        deltas       = keras.backend.variable(targets[0][:, :, :4])
        boxes        = backend.bbox_transform_inv(anchors_, deltas)
        boxes        = keras.backend.eval(boxes)

        for b in range(args.batch_size):
            # compute image back to [0, 255]
            image = inputs[b, ...]
            image -= min(image.flatten())
            image /= max(image.flatten())
            image = (image * 255).astype(np.uint8)

            # draw anchors on the image
            if args.anchors:
                draw_boxes(image, anchors[b, anchor_state[b, :] == 1, :], (255, 255, 0), thickness=1)

            # draw annotations on the image
            if args.annotations:
                # draw annotations in red
                draw_annotations(image, annotations[b], color=(0, 0, 255), generator=generator)

                # draw regressed anchors in green to override most red annotations
                # result is that annotations without anchors are red, with anchors are green
                draw_boxes(image, boxes[b, anchor_state[b, :] == 1, :], (0, 255, 0))

            cv2.imshow('Image', image)
            if cv2.waitKey() == ord('q'):
                return False

    return True


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generator
    generator = create_generator(args)

    # create the display window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    if args.loop:
        while run(generator, args):
            pass
    else:
        run(generator, args)

if __name__ == '__main__':
    main()
