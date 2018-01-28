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

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..preprocessing.csv_generator import CSVGenerator
from ..utils.transform import random_transform_generator
from ..utils.visualization import draw_annotations


def create_generator(args):
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        # min_rotation=-0.1,
        # max_rotation=0.1,
        # min_translation=(-0.1, -0.1),
        # max_translation=(0.1, 0.1),
        # min_shear=-0.1,
        # max_shear=0.1,
        # min_scaling=(0.9, 0.9),
        # max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        generator = CocoGenerator(
            args.coco_path,
            args.coco_set,
            transform_generator=transform_generator
        )
    elif args.dataset_type == 'pascal':
        generator = PascalVocGenerator(
            args.pascal_path,
            args.pascal_set,
            transform_generator=transform_generator
        )
    elif args.dataset_type == 'csv':
        generator = CSVGenerator(
            args.annotations,
            args.classes,
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

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes',     help='Path to a CSV file containing class label mapping.')

    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--annotations', help='Show annotations on the image.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generator
    generator = create_generator(args)

    # create the display window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    # display images, one at a time
    for i in range(generator.size()):
        # load the data
        image       = generator.load_image(i)
        annotations = generator.load_annotations(i)

        # apply random transformations
        if args.random_transform:
            image, annotations = generator.random_transform_group_entry(image, annotations)

        # resize the image and annotations
        if args.resize:
            image, image_scale = generator.resize_image(image)
            if args.annotations:
                annotations[:, :4] *= image_scale

        # draw annotations on the image
        if args.annotations:
            draw_annotations(image, annotations, generator=generator)

        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            break

if __name__ == '__main__':
    main()
