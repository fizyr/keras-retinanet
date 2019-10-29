import tensorflow

# parse the tensorflow version
tf_version = [int(i) for i in tensorflow.__version__.split('.')]
if tf_version[0] >= 2:
    # Disable Tensorflow 2.x behavior as we experience issues with it.
    # see: https://www.tensorflow.org/api_docs/python/tf/compat/v1/disable_v2_behavior
    tensorflow.compat.v1.disable_v2_behavior()
    # for cases of using `x.numpy()`
    # Aiming: NotImplementedError: numpy() is only available when eager execution is enabled.
    # see: https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_eager_execution
    tensorflow.compat.v1.enable_eager_execution()

from .dynamic import *  # noqa: F401,F403
from .common import *   # noqa: F401,F403
