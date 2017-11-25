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

from __future__ import print_function

from keras_retinanet.utils.anchors import compute_overlap

import numpy as np
import os

import cv2
import pickle


def _compute_ap(recall, precision):
    # code originally from https://github.com/rbgirshick/py-faster-rcnn

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100):
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        image = generator.load_image(i)
        image = generator.preprocess_image(image)
        image, scale = generator.resize_image(image)

        # run network
        _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

        # correct boxes for image scale
        detections[0, :, :4] /= scale

        # select scores from detections
        scores = detections[0, :, 4:]

        # select indices which have a score above the threshold
        indices = np.where(detections[0, :, 4:] > score_threshold)

        # select those scores
        scores = scores[indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = detections[0, indices[0][scores_sort], :4]
        image_scores     = np.expand_dims(detections[0, indices[0][scores_sort], 4 + indices[1][scores_sort]], axis=1)
        image_detections = np.append(image_boxes, image_scores, axis=1)
        image_predicted_labels = indices[1][scores_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_predicted_labels == label, :]

        #print([generator.label_to_name(l) for l in image_predicted_labels])
        print('{}/{}'.format(i, generator.size()), end='\r')

    return all_detections


def _get_annotations(generator):
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        #print([generator.label_to_name(l) for l in annotations[:, 4]])
        print('{}/{}'.format(i, generator.size()), end='\r')

    return all_annotations


def evaluate(generator, model, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    #all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections)
    #all_annotations    = _get_annotations(generator)
    average_precisions = np.zeros((0,))

    all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    #pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    #pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        false_positives    = np.cumsum(false_positives)
        true_positives     = np.cumsum(true_positives)
        recall             = true_positives / num_annotations
        precision          = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        average_precision  = _compute_ap(recall, precision)
        average_precisions = np.append(average_precisions, average_precision)

        print(generator.label_to_name(label), average_precision)

    print('mAP: {}'.format(average_precisions.mean()))
