import numpy as np
import configparser
from tensorflow import keras

from keras_retinanet.utils.anchors import anchors_for_shape, AnchorParameters
from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters


def test_config_read():
    config = read_config_file('tests/test-data/config/config.ini')
    assert 'anchor_parameters' in config
    assert 'sizes' in config['anchor_parameters']
    assert 'strides' in config['anchor_parameters']
    assert 'ratios' in config['anchor_parameters']
    assert 'scales' in config['anchor_parameters']
    assert config['anchor_parameters']['sizes']   == '32 64 128 256 512'
    assert config['anchor_parameters']['strides'] == '8 16 32 64 128'
    assert config['anchor_parameters']['ratios']  == '0.5 1 2 3'
    assert config['anchor_parameters']['scales']  == '1 1.2 1.6'


def create_anchor_params_config():
    config = configparser.ConfigParser()
    config['anchor_parameters'] = {}
    config['anchor_parameters']['sizes']   = '32 64 128 256 512'
    config['anchor_parameters']['strides'] = '8 16 32 64 128'
    config['anchor_parameters']['ratios']  = '0.5 1'
    config['anchor_parameters']['scales']  = '1 1.2 1.6'

    return config


def test_parse_anchor_parameters():
    config = create_anchor_params_config()
    anchor_params_parsed = parse_anchor_parameters(config)

    sizes   = [32, 64, 128, 256, 512]
    strides = [8, 16, 32, 64, 128]
    ratios  = np.array([0.5, 1], keras.backend.floatx())
    scales  = np.array([1, 1.2, 1.6], keras.backend.floatx())

    assert sizes   == anchor_params_parsed.sizes
    assert strides == anchor_params_parsed.strides
    np.testing.assert_equal(ratios, anchor_params_parsed.ratios)
    np.testing.assert_equal(scales, anchor_params_parsed.scales)


def test_anchors_for_shape_dimensions():
    sizes   = [32, 64, 128]
    strides = [8, 16, 32]
    ratios  = np.array([0.5, 1, 2, 3], keras.backend.floatx())
    scales  = np.array([1, 1.2, 1.6], keras.backend.floatx())
    anchor_params = AnchorParameters(sizes, strides, ratios, scales)

    pyramid_levels = [3, 4, 5]
    image_shape    = (64, 64)
    all_anchors    = anchors_for_shape(image_shape, pyramid_levels=pyramid_levels, anchor_params=anchor_params)

    assert all_anchors.shape == (1008, 4)


def test_anchors_for_shape_values():
    sizes   = [12]
    strides = [8]
    ratios  = np.array([1, 2], keras.backend.floatx())
    scales  = np.array([1, 2], keras.backend.floatx())
    anchor_params = AnchorParameters(sizes, strides, ratios, scales)

    pyramid_levels = [3]
    image_shape    = (16, 16)
    all_anchors    = anchors_for_shape(image_shape, pyramid_levels=pyramid_levels, anchor_params=anchor_params)

    # using almost_equal for floating point imprecisions
    np.testing.assert_almost_equal(all_anchors[0, :], [
        strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[1, :], [
        strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[2, :], [
        strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[3, :], [
        strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[4, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[5, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[6, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[7, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[8, :], [
        strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[9, :], [
        strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[10, :], [
        strides[0] / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[11, :], [
        strides[0] / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        strides[0] / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[12, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[13, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[0])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[0])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[14, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[0] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
    np.testing.assert_almost_equal(all_anchors[15, :], [
        strides[0] * 3 / 2 - (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 - (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] / np.sqrt(ratios[1])) / 2,
        strides[0] * 3 / 2 + (sizes[0] * scales[1] * np.sqrt(ratios[1])) / 2,
    ], decimal=6)
