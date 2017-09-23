import setuptools

setuptools.setup(
    name='keras-retinanet',
    version='0.0.1',
    url='https://github.com/delftrobotics/keras-retinanet',
    author='Hans Gaiser',
    author_email='j.c.gaiser@delftrobotics.com',
    maintainer='Hans Gaiser',
    maintainer_email='j.c.gaiser@delftrobotics.com',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet']
)
