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


def random_transform(
    min_rotation,
    max_rotation,
    min_translation,
    max_translation,
    min_shear,
    max_shear,
    min_scaling,
    max_scaling,
    flip_x_chance,
    flip_y_chance,
    prng = DEFAULT_PRNG
):
    """ Create a random transformation.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    # Arguments
        min_rotation:    The minimum rotation for the transform as scalar.
        max_rotation:    The maximum rotation for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear:       The minimum shear for the transform as scalar.
        max_shear:       The maximum shear for the transform as scalar.
        min_scaling:     The minimum scaling for the transform as 2D column vector.
        max_scaling:     The maximum scaling for the transform as 2D column vector.
        flip_x_chance:   The chance (0 to 1) that a transform will contain a flip along X direction.
        flip_y_chance:   The chance (0 to 1) that a transform will contain a flip along Y direction.
        prng:            The pseudo-random number generator to use.
    """
    result = random_rotation(min_rotation, max_rotation, prng)
    result = np.dot(result, random_translation(min_translation, max_translation, prng))
    result = np.dot(result, random_shear(min_shear, max_shear, prng))
    result = np.dot(result, random_scaling(min_scaling, max_scaling, prng))
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    result = np.dot(result, scaling(vec2(-1 if flip_x else 1, -1 if flip_y else 1)))
    return result


def random_transform_generator(*args, **kwargs):
    """ Create a random transform generator with the same arugments as `random_transform`. """
    while True:
        yield random_transform(*args, **kwargs)


def _translate_image_data_generator_params(image_data_generator):
    """ Translate the properties of a Keras ImageDataGenerator to keyword arguments for random_transform() """


def random_transform_from_image_data_generator(image_data_generator, prng = DEFAULT_PRNG):
    """ Create a random transform using the same parameters as a Keras ImageDataGenerator.

    Note that the image dimensions are unknown at this points,
    so the transform origin should be modified to the image center before using it.
    Additionally, the translation is relative to the image size.
    You can use `transform_image` to fix these details for you.

    # Arguments
        image_data_generator: The Keras ImageDataGenerator to mimick.
        prng:                 The speudo-random number generator to use.
    """
    rotation    = image_data_generator.rotation_range
    translation = np.array([[image_data_generator.width_shift_range], [image_data_generator.height_shift_range]])
    shear       = image_data_generator.shear_range
    min_zoom    = image_data_generator.zoom_range[0]
    max_zoom    = image_data_generator.zoom_range[1]
    flip_x      = 0.5 if image_data_generator.horizontal_flip else 0
    flip_y      = 0.5 if image_data_generator.vertical_flip else 0

    return random_transform(
        min_rotation    = -rotation,
        max_rotation    = +rotation,
        min_translation = -translation,
        max_translation = +translation,
        min_shear       = -shear,
        max_shear       = +shear,
        min_scaling     = vec2(min_zoom, min_zoom),
        max_scaling     = vec2(max_zoom, max_zoom),
        flip_x_chance   = flip_x,
        flip_y_chance   = flip_y,
        prng            = prng
    )


def random_transform_generator_from_image_data_generator(image_data_generator, prng = DEFAULT_PRNG):
    """ Create a random transform generator that mimicks a Keras ImageDataGenerator.
    # Arguments
        image_data_generator: The Keras ImageDataGenerator to mimick.
        prng:                 The speudo-random number generator to use.
    """
    while True:
        yield random_transform_from_image_data_generator(image_data_generator, prng)
