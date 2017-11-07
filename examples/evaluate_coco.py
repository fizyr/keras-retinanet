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

import keras
import keras.preprocessing.image
import keras_retinanet
import keras_retinanet.preprocessing.coco

from keras_retinanet.preprocessing.coco import CocoIterator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json
import cv2
import numpy as np
import os
import argparse

import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_model(weights='imagenet'):
    image = keras.layers.Input((None, None, 3))
    return keras_retinanet.models.ResNet50RetinaNet(image, num_classes=90, weights=weights)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for COCO object detection.')
    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('coco_path', help='Path to COCO directory (ie. /tmp/COCO).')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--set', help='Name of the set file to evaluate (defaults to val2017).', default='val2017')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Creating model, this may take a second...')
    model = create_model(weights=args.model)

    # create image data generator object
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for testing data
    test_generator = CocoIterator(
        args.coco_path,
        args.set,
        test_image_data_generator,
    )

    # start collecting results
    results = []
    image_ids = []
    for i in range(len(test_generator.image_ids)):
        image_data = test_generator.load_image(i)
        if image_data is None:
            # some images fail to load due to missing annotations
            continue

        # run network
        _, _, detections = model.predict_on_batch(image_data['image_batch'])

        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(image_data['image'].shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(image_data['image'].shape[0], detections[:, :, 3])

        # correct boxes for image scale
        detections[0, :, :4] /= image_data['image_scale']

        # change to (x, y, w, h) (MS COCO standard)
        detections[:, :, 2] -= detections[:, :, 0]
        detections[:, :, 3] -= detections[:, :, 1]

        # compute predicted labels and scores
        for detection in detections[0, ...]:
            positive_labels = np.where(detection[4:] > args.score_threshold)[0]

            # append detections for each positively labeled class
            for label in positive_labels:
                image_result = {
                    'image_id'    : image_data['coco_index'],
                    'category_id' : int(label) + 1,  # MS COCO starts labels from 1, we start from 0
                    'score'       : float(detection[4 + label]),
                    'bbox'        : (detection[:4]).tolist(),
                }

                # append detection to results
                results.append(image_result)

        # append image to list of processed images
        image_ids.append(image_data['coco_index'])

        # print progress
        print('{}/{}'.format(i, len(test_generator.image_ids)), end='\r')

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(args.set), 'w'), indent=4)
    json.dump(image_ids, open('{}_processed_image_ids.json'.format(args.set), 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_true = test_generator.coco
    coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(args.set))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
