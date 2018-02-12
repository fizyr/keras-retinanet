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

class VocPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(VocPlugin, self).__init__()
        self.dataset_type = "pascal"


    def register_parser_args(self, subparsers):
        pascal_parser = subparsers.add_parser(self.dataset_type)
        pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

        return pascal_parser
