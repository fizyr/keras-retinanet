import numpy as np
from numpy.testing import assert_almost_equal
from math import pi

from keras_retinanet.utils.transform import (
    colvec,
    transform_aabb,
    rotation, random_rotation,
    translation, random_translation,
    scaling, random_scaling,
    shear, random_shear,
    random_flip,
    random_transform,
    random_transform_generator,
    change_transform_origin,
)


def test_colvec():
    assert np.array_equal(colvec(0), np.array([[0]]))
    assert np.array_equal(colvec(1, 2, 3), np.array([[1], [2], [3]]))
    assert np.array_equal(colvec(-1, -2), np.array([[-1], [-2]]))


def test_rotation():
    assert_almost_equal(colvec( 1,  0, 1), rotation(0.0 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec( 0,  1, 1), rotation(0.5 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec(-1,  0, 1), rotation(1.0 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec( 0, -1, 1), rotation(1.5 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec( 1,  0, 1), rotation(2.0 * pi).dot(colvec(1, 0, 1)))

    assert_almost_equal(colvec( 0,  1, 1), rotation(0.0 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec(-1,  0, 1), rotation(0.5 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec( 0, -1, 1), rotation(1.0 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec( 1,  0, 1), rotation(1.5 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec( 0,  1, 1), rotation(2.0 * pi).dot(colvec(0, 1, 1)))


def test_random_rotation():
    prng = np.random.RandomState(0)
    for i in range(100):
        assert_almost_equal(1, np.linalg.det(random_rotation(-i, i, prng)))


def test_translation():
    assert_almost_equal(colvec( 1,  2, 1), translation(colvec( 0,  0)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec( 4,  6, 1), translation(colvec( 3,  4)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(-2, -2, 1), translation(colvec(-3, -4)).dot(colvec(1, 2, 1)))


def assert_is_translation(transform, min, max):
    assert transform.shape == (3, 3)
    assert np.array_equal(transform[:, 0:2], np.eye(3, 2))
    assert transform[2, 2] == 1
    assert np.greater_equal(transform[0:2, 2], min).all()
    assert np.less(         transform[0:2, 2], max).all()


def test_random_translation():
    prng = np.random.RandomState(0)
    min = (-10, -20)
    max = (20, 10)
    for i in range(100):
        assert_is_translation(random_translation(min, max, prng), min, max)


def test_shear():
    assert_almost_equal(colvec( 1,  2, 1), shear(0.0 * pi).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(-1,  0, 1), shear(0.5 * pi).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec( 1, -2, 1), shear(1.0 * pi).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec( 3,  0, 1), shear(1.5 * pi).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec( 1,  2, 1), shear(2.0 * pi).dot(colvec(1, 2, 1)))


def assert_is_shear(transform):
    assert transform.shape == (3, 3)
    assert np.array_equal(transform[:, 0], [1, 0, 0])
    assert np.array_equal(transform[:, 2], [0, 0, 1])
    assert transform[2, 1] == 0
    # sin^2 + cos^2 == 1
    assert_almost_equal(1, transform[0, 1] ** 2 + transform[1, 1] ** 2)


def test_random_shear():
    prng = np.random.RandomState(0)
    for i in range(100):
        assert_is_shear(random_shear(-pi, pi, prng))


def test_scaling():
    assert_almost_equal(colvec(1.0, 2, 1), scaling(colvec(1.0, 1.0)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(0.0, 2, 1), scaling(colvec(0.0, 1.0)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(1.0, 0, 1), scaling(colvec(1.0, 0.0)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(0.5, 4, 1), scaling(colvec(0.5, 2.0)).dot(colvec(1, 2, 1)))


def assert_is_scaling(transform, min, max):
    assert transform.shape == (3, 3)
    assert np.array_equal(transform[2, :], [0, 0, 1])
    assert np.array_equal(transform[:, 2], [0, 0, 1])
    assert transform[1, 0] == 0
    assert transform[0, 1] == 0
    assert np.greater_equal(np.diagonal(transform)[:2], min).all()
    assert np.less(         np.diagonal(transform)[:2], max).all()


def test_random_scaling():
    prng = np.random.RandomState(0)
    min = (0.1, 0.2)
    max = (20, 10)
    for i in range(100):
        assert_is_scaling(random_scaling(min, max, prng), min, max)


def assert_is_flip(transform):
    assert transform.shape == (3, 3)
    assert np.array_equal(transform[2, :], [0, 0, 1])
    assert np.array_equal(transform[:, 2], [0, 0, 1])
    assert transform[1, 0] == 0
    assert transform[0, 1] == 0
    assert abs(transform[0, 0]) == 1
    assert abs(transform[1, 1]) == 1


def test_random_flip():
    prng = np.random.RandomState(0)
    for i in range(100):
        assert_is_flip(random_flip(0.5, 0.5, prng))


def test_random_transform():
    prng = np.random.RandomState(0)
    for i in range(100):
        transform = random_transform(prng=prng)
        assert np.array_equal(transform, np.identity(3))

    for i, transform in zip(range(100), random_transform_generator(prng=np.random.RandomState())):
        assert np.array_equal(transform, np.identity(3))


def test_transform_aabb():
    assert np.array_equal([1, 2, 3, 4], transform_aabb(np.identity(3), [1, 2, 3, 4]))
    assert_almost_equal([-3, -4, -1, -2], transform_aabb(rotation(pi),        [1, 2, 3, 4]))
    assert_almost_equal([ 2,  4,  4,  6], transform_aabb(translation([1, 2]), [1, 2, 3, 4]))


def test_change_transform_origin():
    prng = np.random.RandomState(0)
    assert np.array_equal(change_transform_origin(translation([3, 4]), [1, 2]), translation([3, 4]))
    assert_almost_equal(colvec(1, 2, 1), change_transform_origin(rotation(pi), [1, 2]).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(0, 0, 1), change_transform_origin(rotation(pi), [1, 2]).dot(colvec(2, 4, 1)))
    assert_almost_equal(colvec(0, 0, 1), change_transform_origin(scaling([0.5, 0.5]), [-2, -4]).dot(colvec(2, 4, 1)))
