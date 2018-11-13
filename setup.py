import setuptools
from setuptools.extension import Extension

class custom_wrapper(object):
    def __new__(cls, *args, **kwargs):
        from setuptools.command.build_ext import build_ext
        class custom_build_ext(build_ext, object):
            def run(self):
                import numpy as np
                self.include_dirs.append(np.get_include())
                build_ext.run(self)
        return custom_build_ext(*args, **kwargs)

extensions = [
    Extension(
        'keras_retinanet.utils.compute_overlap',
        ['keras_retinanet/utils/compute_overlap.pyx']
    ),
]

setuptools.setup(
    name             = 'keras-retinanet',
    version          = '0.5.0',
    description      = 'Keras implementation of RetinaNet object detection.',
    url              = 'https://github.com/fizyr/keras-retinanet',
    author           = 'Hans Gaiser',
    author_email     = 'h.gaiser@fizyr.com',
    maintainer       = 'Hans Gaiser',
    maintainer_email = 'h.gaiser@fizyr.com',
    cmdclass         = {'build_ext': custom_wrapper},
    packages         = setuptools.find_packages(),
    install_requires = ['keras', 'keras-resnet', 'six', 'scipy', 'cython', 'Pillow', 'opencv-python', 'progressbar2'],
    entry_points     = {
        'console_scripts': [
            'retinanet-train=keras_retinanet.bin.train:main',
            'retinanet-evaluate=keras_retinanet.bin.evaluate:main',
            'retinanet-debug=keras_retinanet.bin.debug:main',
            'retinanet-convert-model=keras_retinanet.bin.convert_model:main',
        ],
    },
    ext_modules    = extensions,
    setup_requires = ["cython>=0.28", "numpy>=1.14.0"]
)
