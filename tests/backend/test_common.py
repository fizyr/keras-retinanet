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

import numpy as np
from tensorflow import keras
import keras_retinanet.backend


def test_bbox_transform_inv():
    boxes = np.array([[
        [100, 100, 200, 200],
        [100, 100, 300, 300],
        [100, 100, 200, 300],
        [100, 100, 300, 200],
        [80,  120, 200, 200],
        [80,  120, 300, 300],
        [80,  120, 200, 300],
        [80,  120, 300, 200],
    ]])
    boxes = keras.backend.variable(boxes)

    deltas = np.array([[
        [0   , 0  , 0   , 0   ],
        [0   , 0.1, 0   , 0   ],
        [-0.3, 0  , 0   , 0   ],
        [0.2 , 0.2, 0   , 0   ],
        [0   , 0  , 0.1 , 0   ],
        [0   , 0  , 0   , -0.3],
        [0   , 0  , 0.2 , 0.2 ],
        [0.1 , 0.2, -0.3, 0.4 ],
    ]])
    deltas = keras.backend.variable(deltas)

    expected = np.array([[
        [100  , 100  , 200   , 200  ],
        [100  , 104  , 300   , 300  ],
        [ 94  , 100  , 200   , 300  ],
        [108  , 104  , 300   , 200  ],
        [ 80  , 120  , 202.4 , 200  ],
        [ 80  , 120  , 300   , 289.2],
        [ 80  , 120  , 204.8 , 307.2],
        [ 84.4, 123.2, 286.8 , 206.4]
    ]])

    result = keras_retinanet.backend.bbox_transform_inv(boxes, deltas)
    result = keras.backend.eval(result)

    np.testing.assert_array_almost_equal(result, expected, decimal=2)


def test_shift():
    shape  = (2, 3)
    stride = 8

    anchors = np.array([
        [-8,  -8,  8,  8],
        [-16, -16, 16, 16],
        [-12, -12, 12, 12],
        [-12, -16, 12, 16],
        [-16, -12, 16, 12]
    ], dtype=keras.backend.floatx())

    expected = [
        # anchors for (0, 0)
        [4 - 8,  4 - 8,  4 + 8,  4 + 8],
        [4 - 16, 4 - 16, 4 + 16, 4 + 16],
        [4 - 12, 4 - 12, 4 + 12, 4 + 12],
        [4 - 12, 4 - 16, 4 + 12, 4 + 16],
        [4 - 16, 4 - 12, 4 + 16, 4 + 12],

        # anchors for (0, 1)
        [12 - 8,  4 - 8,  12 + 8,  4 + 8],
        [12 - 16, 4 - 16, 12 + 16, 4 + 16],
        [12 - 12, 4 - 12, 12 + 12, 4 + 12],
        [12 - 12, 4 - 16, 12 + 12, 4 + 16],
        [12 - 16, 4 - 12, 12 + 16, 4 + 12],

        # anchors for (0, 2)
        [20 - 8,  4 - 8,  20 + 8,  4 + 8],
        [20 - 16, 4 - 16, 20 + 16, 4 + 16],
        [20 - 12, 4 - 12, 20 + 12, 4 + 12],
        [20 - 12, 4 - 16, 20 + 12, 4 + 16],
        [20 - 16, 4 - 12, 20 + 16, 4 + 12],

        # anchors for (1, 0)
        [4 - 8,  12 - 8,  4 + 8,  12 + 8],
        [4 - 16, 12 - 16, 4 + 16, 12 + 16],
        [4 - 12, 12 - 12, 4 + 12, 12 + 12],
        [4 - 12, 12 - 16, 4 + 12, 12 + 16],
        [4 - 16, 12 - 12, 4 + 16, 12 + 12],

        # anchors for (1, 1)
        [12 - 8,  12 - 8,  12 + 8,  12 + 8],
        [12 - 16, 12 - 16, 12 + 16, 12 + 16],
        [12 - 12, 12 - 12, 12 + 12, 12 + 12],
        [12 - 12, 12 - 16, 12 + 12, 12 + 16],
        [12 - 16, 12 - 12, 12 + 16, 12 + 12],

        # anchors for (1, 2)
        [20 - 8,  12 - 8,  20 + 8,  12 + 8],
        [20 - 16, 12 - 16, 20 + 16, 12 + 16],
        [20 - 12, 12 - 12, 20 + 12, 12 + 12],
        [20 - 12, 12 - 16, 20 + 12, 12 + 16],
        [20 - 16, 12 - 12, 20 + 16, 12 + 12],
    ]

    result = keras_retinanet.backend.shift(shape, stride, anchors)
    result = keras.backend.eval(result)

    np.testing.assert_array_equal(result, expected)
