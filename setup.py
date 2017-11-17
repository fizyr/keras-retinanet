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
    install_requires=['keras', 'keras-resnet', 'six']
)
