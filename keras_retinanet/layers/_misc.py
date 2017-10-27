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
import keras_retinanet

import numpy as np


class Anchors(keras.layers.Layer):
    def __init__(self, anchor_size, anchor_stride, anchor_ratios=None, anchor_scales=None, *args, **kwargs):
        self.anchor_size   = anchor_size
        self.anchor_stride = anchor_stride
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales

        self.num_anchors = len(anchor_ratios) * len(anchor_scales)
        self.anchors     = keras.backend.variable(keras_retinanet.preprocessing.anchors.generate_anchors(
            base_size=anchor_size,
            ratios=anchor_ratios,
            scales=anchor_scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)[1:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = keras_retinanet.backend.shift(features_shape, self.anchor_stride, self.anchors)
        anchors = keras.backend.expand_dims(anchors, axis=0)

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        return {
            'anchor_size': self.anchor_size,
            'anchor_stride' : self.anchor_stride,
            'anchor_ratios' : self.anchor_ratios,
            'anchor_scales' : self.anchor_scales,
        }


class TensorReshape(keras.layers.Layer):
    """ Nearly identical to keras.layers.Reshape, but allows reshaping tensors of unknown shape.

    # Arguments
        target_shape: Target shape of the input.
    """
    def __init__(self, target_shape, *args, **kwargs):
        self.target_shape = tuple(target_shape)
        super(TensorReshape, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return keras.backend.reshape(inputs, (keras.backend.shape(inputs)[0],) + self.target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Finds and replaces a missing dimension in an output shape.

        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

        # Arguments
            input_shape: original shape of array being reshaped
            output_shape: target shape of the array, with at most
                a single -1 which indicates a dimension that should be
                derived from the input shape.

        # Returns
            The new output shape with a `-1` replaced with its computed value.

        # Raises
            ValueError: if `input_shape` and `output_shape` do not match.
        """
        output_shape = list(output_shape)
        msg = 'total size of new array must be unchanged'

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError('Can only specify one unknown dimension.')
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)

        return tuple(output_shape)

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            # compute target shape when possible
            return (input_shape[0],) + self._fix_unknown_dimension(input_shape[1:], self.target_shape)
        else:
            return (input_shape[0],) + tuple(s if s != -1 else None for s in self.target_shape)

    def get_config(self):
        return {'target_shape': self.target_shape}


class NonMaximumSuppression(keras.layers.Layer):
    def __init__(self, num_classes, nms_threshold=0.4, top_k=1000, max_boxes=300, *args, **kwargs):
        self.num_classes   = num_classes
        self.nms_threshold = nms_threshold
        self.top_k         = top_k
        self.max_boxes     = max_boxes
        super(NonMaximumSuppression, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        boxes, classification, detections = inputs

        # TODO: support batch size > 1.
        boxes          = boxes[0]
        classification = classification[0]
        detections     = detections[0]

        scores          = keras.backend.max(classification, axis=1)
        scores, indices = keras_retinanet.backend.top_k(scores, self.top_k, sorted=False)

        boxes          = keras.backend.gather(boxes, indices)
        classification = keras.backend.gather(classification, indices)
        detections     = keras.backend.gather(detections, indices)

        indices = keras_retinanet.backend.non_max_suppression(boxes, scores, max_output_size=self.max_boxes, iou_threshold=self.nms_threshold)

        detections = keras.backend.gather(detections, indices)
        return keras.backend.expand_dims(detections, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[2][0], None, input_shape[2][2])

    def get_config(self):
        return {
            'num_classes'   : self.num_classes,
            'nms_threshold' : self.nms_threshold,
            'top_k'         : self.top_k,
            'max_boxes'     : self.max_boxes,
        }


class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return keras_retinanet.backend.resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return keras_retinanet.backend.bbox_transform_inv(anchors, regression)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
