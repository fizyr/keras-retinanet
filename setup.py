import setuptools

setuptools.setup(
    name='keras-retinanet',
    version='0.0.1',
    url='https://github.com/fizyr/keras-retinanet',
    author='Hans Gaiser',
    author_email='h.gaiser@fizyr.com',
    maintainer='Hans Gaiser',
    maintainer_email='h.gaiser@fizyr.com',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet', 'six'],
    entry_points = {
        'console_scripts': [
            'retinanet-train=keras_retinanet.bin.train:main',
            'retinanet-evaluate-coco=keras_retinanet.bin.evaluate_coco:main',
            'retinanet-evaluate=keras_retinanet.bin.evaluate:main',
        ],
    }
)
