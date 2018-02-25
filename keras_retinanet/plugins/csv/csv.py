"""
Copyright 2017-2018 Ashley Williamson (https://inp.io)

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

import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

from keras_retinanet.preprocessing.csv_generator import CSVGenerator


class CSVPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(CSVPlugin, self).__init__()

    def register_parser_args(self, subparser):
        subparser.add_argument('annotations',       help='Path to a CSV file containing annotations for training.')
        subparser.add_argument('classes',           help='Path to a CSV file containing class label mapping.')

    def create_generator(self, args, **kwargs):
        return CSVGenerator(
            args.annotations,
            args.classes,
            **kwargs
        )
