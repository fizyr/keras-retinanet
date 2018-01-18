from __future__ import print_function

import keras
import sys

minimum_keras_version = 2, 1, 3


def keras_version():
    return tuple(map(int, keras.__version__.split('.')))


def keras_version_ok():
    return keras_version() >= minimum_keras_version


def assert_keras_version():
    detected = keras.__version__
    required = '.'.join(map(str, minimum_keras_version))
    assert(keras_version() >= minimum_keras_version), 'You are using keras version {}. The minimum required version is {}.'.format(detected, required)


def check_keras_version():
    try:
        assert_keras_version()
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
