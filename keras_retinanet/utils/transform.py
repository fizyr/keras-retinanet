import numpy as np

DEFAULT_PRNG = np.random


def _random_vector(min, max, prng = DEFAULT_PRNG):
    """ Construct a random column vector between min and max.
    # Arguments
        min: the minimum value for each component
        max: the maximum value for each component
    """
    width = np.array(max) - np.array(min)
    return prng.uniform(0, 1, (2, 1)) * width + min


def rotation(angle):
    """ Construct a homogeneous 2D rotation matrix.
    # Arguments
        angle: the angle in radians
    # Returns
        the rotation matrix as 3 by 3 numpy array
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_rotation(min, max, prng = DEFAULT_PRNG):
    """ Construct a random rotation between -max and max.
    # Arguments
        max: the maximum absolute angle in radians
    # Returns
        a homogeneous 3 by 3 rotation matrix
    """
    return rotation(prng.uniform(min, max))


def translation(translation):
    """ Construct a homogeneous 2D translation matrix.
    # Arguments
        translation: the translation as column vector
    # Returns
        the translation matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, 0, translation[0, 0]],
        [0, 1, translation[1, 0]],
        [0, 0, 1]
    ])


def random_translation(min, max, prng = DEFAULT_PRNG):
    """ Construct a random 2D translation between min and max.
    # Arguments
        min: the minumum translation for each dimension
        max: the maximum translation for each dimension
    # Returns
        a homogeneous 3 by 3 translation matrix
    """
    assert min.shape == (2, 1)
    assert max.shape == (2, 1)
    return translation(_random_vector(min, max, prng))


def shear(amount):
    """ Construct a homogeneous 2D shear matrix.
    # Arguments
        amount: the shear amount
    # Returns
        the shear matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, -np.sin(amount), 0],
        [0,  np.cos(amount), 0],
        [0, 0, 1]
    ])


def random_shear(min, max, prng = DEFAULT_PRNG):
    """ Construct a random 2D shear matrix with shear angle between -max and max.
    # Arguments
        amount: the max shear amount
    # Returns
        a homogeneous 3 by 3 shear matrix
    """
    return shear(prng.uniform(min, max))


def scaling(factor):
    """ Construct a homogeneous 2D scaling matrix.
    # Arguments
        factor: a 2D vector for X and Y scaling
    # Returns
        the zoom matrix as 3 by 3 numpy array
    """
    return np.array([
        [factor[0, 0], 0, 0],
        [0, factor[1, 0], 0],
        [0, 0, 1]
    ])


def random_scaling(min, max, prng = DEFAULT_PRNG):
    """ Construct a random 2D scale matrix between -max and max.
    # Arguments
        factor: a 2D vector for maximum X and Y scaling
    # Returns
        a homogeneous 3 by 3 scaling matrix
    """
    assert min.shape == (2, 1)
    assert max.shape == (2, 1)
    return scaling(_random_vector(min, max, prng))


def transform_around(transform, center):
    """ Get a transform applying the given transform with a different origin.
    # Arguments:
        transform: the transformation matrix
        center: the origin of the transformation
    # Return:
        translate(center) * transform * translate(-center)
    """
    return np.dot(np.dot(translation(center), transform), translation(-center))
