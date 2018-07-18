import setuptools
from setuptools.extension import Extension
import numpy as np

extensions = [
    Extension(
        'keras_retinanet.utils.compute_overlap',
        ['keras_retinanet/utils/compute_overlap.pyx'],
        include_dirs=[np.get_include()]
    ),
]

setuptools.setup(
    name             = 'keras-retinanet',
    version          = '0.4.1',
    description      = 'Keras implementation of RetinaNet object detection.',
    url              = 'https://github.com/fizyr/keras-retinanet',
    author           = 'Hans Gaiser',
    author_email     = 'h.gaiser@fizyr.com',
    maintainer       = 'Hans Gaiser',
    maintainer_email = 'h.gaiser@fizyr.com',
    packages         = setuptools.find_packages(),
    install_requires = ['keras', 'keras-resnet', 'six', 'scipy', 'cython'],
    entry_points     = {
        'console_scripts': [
            'retinanet-train=keras_retinanet.bin.train:main',
            'retinanet-evaluate=keras_retinanet.bin.evaluate:main',
            'retinanet-debug=keras_retinanet.bin.debug:main',
            'retinanet-convert-model=keras_retinanet.bin.convert_model:main',
        ],
    },
    ext_modules    = extensions,
    setup_requires = ["cython>=0.28"]
)
