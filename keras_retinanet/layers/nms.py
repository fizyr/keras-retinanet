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
from .. import backend


class NonMaximumSuppression(keras.layers.Layer):
    def __init__(self, nms_threshold=0.5, score_threshold=0.05, max_boxes=300, *args, **kwargs):
        self.nms_threshold   = nms_threshold
        self.score_threshold = score_threshold
        self.max_boxes       = max_boxes
        super(NonMaximumSuppression, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        # TODO: support batch size > 1.
        boxes           = inputs[0][0]
        classification  = inputs[1][0]
        indices         = backend.range(keras.backend.shape(classification)[0])
        selected_scores = []

        # perform per class NMS
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]

            # threshold based on score
            score_indices = backend.where(keras.backend.greater(scores, self.score_threshold))
            score_indices = keras.backend.cast(score_indices, 'int32')
            boxes_        = backend.gather_nd(boxes, score_indices)
            scores        = keras.backend.gather(scores, score_indices)[:, 0]

            # perform NMS
            nms_indices = backend.non_max_suppression(boxes_, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold)

            # filter set of original indices
            selected_indices = keras.backend.gather(score_indices, nms_indices)

            # mask original classification column, setting all suppressed values to 0
            scores = keras.backend.gather(scores, nms_indices)
            scores = backend.scatter_nd(selected_indices, scores, keras.backend.shape(classification[:, c]))
            scores = keras.backend.expand_dims(scores, axis=1)

            selected_scores.append(scores)

        # reconstruct the (suppressed) classification scores
        classification = keras.backend.concatenate(selected_scores, axis=1)

        return keras.backend.expand_dims(classification, axis=0)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(NonMaximumSuppression, self).get_config()
        config.update({
            'nms_threshold'   : self.nms_threshold,
            'score_threshold' : self.score_threshold,
            'max_boxes'       : self.max_boxes,
        })

        return config
