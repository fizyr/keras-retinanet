"""
Copyright 2018 Fizyr (https://fizyr.com)
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
from yapsy.PluginManager import PluginManager
from warnings import warn


def load_plugins(extension, paths, check=None):
    manager = PluginManager()
    manager.getPluginLocator().setPluginPlaces(paths)
    manager.getPluginLocator().setPluginInfoExtension(extension)
    manager.collectPlugins()

    result = {}
    for plugin in manager.getAllPlugins():
        plugin.activate()
        name   = plugin.name
        plugin = plugin.plugin_object
        if check is not None:
            if not check(plugin.name, plugin.plugin_object):
                continue
        result[name] = plugin
    return result


def check_dataset_plugin(name, plugin):
    if not hasattr(plugin, 'create_generators'):
        warn("dataset plugin `{}' is missing a `create_generators' function".format(name))
        return False

    if not callable(plugin.create_generators):
        warn("dataset plugin `{}' attribute `create_generators' is not callable".format(name))
        return False

    return True


def load_dataset_plugins(paths):
    return load_plugins('.dataset', paths, check=check_dataset_plugin)


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
