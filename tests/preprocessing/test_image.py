import os
import pytest
from PIL import Image
from keras_retinanet.utils import image
import numpy as np

_STUB_IMG_FNAME = 'stub-image.jpg'


@pytest.fixture(autouse=True)
def run_around_tests(tmp_path):
    """Create a temp image for test"""
    rand_img = np.random.randint(0, 255, (3, 3, 3), dtype='uint8')
    Image.fromarray(rand_img).save(os.path.join(tmp_path, _STUB_IMG_FNAME))
    yield


def test_read_image_bgr(tmp_path):
    stub_image_path = os.path.join(tmp_path, _STUB_IMG_FNAME)

    original_img = np.asarray(Image.open(
        stub_image_path).convert('RGB'))[:, :, ::-1]
    loaded_image = image.read_image_bgr(stub_image_path)

    # Assert images are equal
    np.testing.assert_array_equal(original_img, loaded_image)
