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


def filter_detections(boxes, classification, other=[], nms=True, score_threshold=0.05, max_detections=300, nms_threshold=0.5):
    """ Filter detections using the boxes and classification values.

    Args
        boxes           : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification  : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other           : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        nms             : Flag to enable/disable non maximum suppression.
        score_threshold : Threshold used to prefilter the boxes with.
        max_detections  : Maximum number of detections to keep.
        nms_threshold   : Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    all_boxes  = []
    all_scores = []
    all_labels = []
    all_other  = []

    # perform per class filtering
    for c in range(int(classification.shape[1])):
        scores = classification[:, c]

        # threshold based on score
        score_indices   = backend.where(keras.backend.greater(scores, score_threshold))
        score_indices   = keras.backend.cast(score_indices, 'int32')
        filtered_boxes  = backend.gather_nd(boxes, score_indices)
        filtered_scores = keras.backend.gather(scores, score_indices)[:, 0]
        filtered_other  = [backend.gather_nd(o, score_indices) for o in other]

        if nms:
            # perform NMS
            nms_indices = backend.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter NMS detections
            filtered_boxes  = keras.backend.gather(filtered_boxes, nms_indices)
            filtered_scores = keras.backend.gather(filtered_scores, nms_indices)
            filtered_other  = [keras.backend.gather(o, nms_indices) for o in filtered_other]

        # labels is a vector of the current class label
        filtered_labels = c * keras.backend.ones((keras.backend.shape(filtered_scores)[0],), dtype='int32')

        # append to lists
        all_boxes.append(filtered_boxes)
        all_scores.append(filtered_scores)
        all_labels.append(filtered_labels)
        all_other.append(filtered_other)

    # concatenate outputs to single tensors
    boxes  = keras.backend.concatenate(all_boxes, axis=0)
    scores = keras.backend.concatenate(all_scores, axis=0)
    labels = keras.backend.concatenate(all_labels, axis=0)
    other  = [keras.backend.concatenate([o[i] for o in all_other], axis=0) for i in range(len(other))]

    # select top k
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))
    boxes               = keras.backend.gather(boxes, top_indices)
    labels              = keras.backend.gather(labels, top_indices)
    other               = [keras.backend.gather(o, top_indices) for o in other]

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes    = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = keras.backend.cast(labels, 'int32')
    other    = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other]

    return [boxes, scores, labels] + other


class FilterDetections(keras.layers.Layer):
    def __init__(
        self,
        nms                 = True,
        nms_threshold       = 0.5,
        score_threshold     = 0.05,
        max_detections      = 300,
        parallel_iterations = 32,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms                 : Flag to enable/disable NMS.
            nms_threshold       : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold     : Threshold used to prefilter the boxes with.
            max_detections      : Maximum number of detections to keep.
            parallel_iterations : Number of batch items to process in parallel.
        """
        self.nms                 = nms
        self.nms_threshold       = nms_threshold
        self.score_threshold     = score_threshold
        self.max_detections      = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms=self.nms,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + input_shape[i][2:]) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                 : self.nms,
            'nms_threshold'       : self.nms_threshold,
            'score_threshold'     : self.score_threshold,
            'max_detections'      : self.max_detections,
            'parallel_iterations' : self.parallel_iterations,
        })

        return config
