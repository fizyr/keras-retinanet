"""
Copyright 2017-2019 Fizyr (https://fizyr.com)

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

import tensorflow as tf


def setup_gpu(gpu_id):
    if gpu_id == 'cpu' or gpu_id == -1:
        tf.config.set_visible_devices([], 'GPU')
        return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU.
        try:
            # Currently, memory growth needs to be the same across GPUs.
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Use only the selcted gpu.
            tf.config.set_visible_devices(gpus[int(gpu_id)], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized.
            print(e)

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
