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
from ..utils.eval import evaluate


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(self, generator, iou_threshold=0.5, score_threshold=0.05, max_detections=100, save_path=None, tensorboard=None, verbose=1):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.verbose         = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        present_classes = 0
        precision = 0
        for label, (average_precision, num_annotations ) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            if num_annotations > 0:
                present_classes += 1
                precision       += average_precision
        self.mean_ap = precision / present_classes

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
