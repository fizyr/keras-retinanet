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

import configparser
import numpy as np
import keras
from ..utils.anchors import AnchorParameters

def read_parameters_file(parameters_path):
    config = configparser.ConfigParser()
    config.read(parameters_path)

    return config

def parse_anchor_parameters(config):
    ratios  = []
    scales  = []
    sizes   = []
    strides = []
    for ratio in config['anchor_parameters']['ratios'].split(' '):
        ratios.append(float(ratio))
    ratios = np.array(ratios, keras.backend.floatx())
    for scale in config['anchor_parameters']['scales'].split(' '):
        scales.append(float(scale))
    scales = np.array(scales, keras.backend.floatx())
    for size in config['anchor_parameters']['sizes'].split(' '):
        sizes.append(int(size))
    for stride in config['anchor_parameters']['strides'].split(' '):
        strides.append(int(stride))

    return AnchorParameters(sizes, strides, ratios, scales)
