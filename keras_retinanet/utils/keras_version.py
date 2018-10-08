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

from __future__ import print_function

import keras
import sys

minimum_keras_version = 2, 2, 0


def keras_version():
    """ Get the Keras version.

    Returns
        tuple of (major, minor, patch).
    """
    return tuple(map(int, keras.__version__.split('.')))


def keras_version_ok():
    """ Check if the current Keras version is higher than the minimum version.
    """
    return keras_version() >= minimum_keras_version


def assert_keras_version():
    """ Assert that the Keras version is up to date.
    """
    detected = keras.__version__
    required = '.'.join(map(str, minimum_keras_version))
    assert(keras_version() >= minimum_keras_version), 'You are using keras version {}. The minimum required version is {}.'.format(detected, required)


def check_keras_version():
    """ Check that the Keras version is up to date. If it isn't, print an error message and exit the script.
    """
    try:
        assert_keras_version()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
