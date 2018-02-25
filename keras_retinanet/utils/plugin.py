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
    for info in manager.getAllPlugins():
        manager.activatePluginByName(info.name)
        name   = info.name
        plugin = info.plugin_object
        if check is not None:
            if not check(name, plugin):
                continue
        result[name] = plugin
    return result


def _check_function(plugin, function_name, plugin_type, plugin_name):
    if not hasattr(plugin, function_name):
        warn("{} plugin `{}' is missing a `{}' function".format(plugin_type, plugin_name, function_name))
        return False

    if not callable(getattr(plugin, function_name)):
        warn("{} plugin `{}' attribute `{}' is not callable".format(plugin_type, plugin_name, function_name))
        return False
    return True


def check_dataset_plugin(name, plugin):
    if not _check_function(plugin, 'create_generator',     'dataset', name):
        return False
    if not _check_function(plugin, 'register_parser_args', 'dataset', name):
        return False
    if not _check_function(plugin, 'check_args', 'dataset', name):
        return False
    return True


def load_dataset_plugins(paths):
    return load_plugins('dataset', paths, check=check_dataset_plugin)


class DatasetPlugin(IPlugin):
    def __init__(self):
        self.dataset_type = None
        super(DatasetPlugin, self).__init__()

    def register_parser_args(self, subparser):
        raise NotImplementedError

    def check_args(self, parsed_args):
        pass

    def create_generator(self, args):
        raise NotImplementedError
