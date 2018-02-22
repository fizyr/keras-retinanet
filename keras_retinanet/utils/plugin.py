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

from yapsy.IPlugin import IPlugin


class DatasetPlugin(IPlugin):
    def __init__(self):
        self.dataset_type = None
        super(DatasetPlugin, self).__init__()

    def register_parser_args(self, subparser):
        raise NotImplementedError

    def check_args(self, parsed_args):
        pass

    def create_generators(self, args):
        raise NotImplementedError