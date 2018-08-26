import keras_retinanet.bin.train
import keras_retinanet.bin.evaluate
from keras_retinanet.bin.train import get_anchors_params
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.anchors import anchors_for_shape

import warnings

def test_csv_generator_anchors():
    anchors_dict = get_anchors_params("tests/test-data/anchors.yaml")
    train_generator = CSVGenerator(
        "tests/test-data/csv/annotations.csv",
        "tests/test-data/csv/classes.csv",
        transform_generator=None,
        batch_size=1,
        image_min_side=512,
        image_max_side=512,
        **anchors_dict
    )

    inputs,targets = train_generator.next()
    regreession_batch,labels_batch = targets
    labels = labels_batch[0]
    image = inputs[0]
    anchors = anchors_for_shape(image.shape,**anchors_dict)
    assert len(labels) == len(anchors)
    
def test_train_generate_anchors_config():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.train.main([
        '--epochs=1',
        '--steps=1',
        '--no-weights',
        '--anchors',
        'tests/test-data/anchors.yaml',
        '--snapshot-path',
        'tests/snapshot',
        'csv',
        'tests/test-data/csv/annotations.csv',
        'tests/test-data/csv/classes.csv',
    ])

def test_evaluate_config_anchors_params():
    # ignore warnings in this test
    warnings.simplefilter('ignore')

    # run training / evaluation
    keras_retinanet.bin.evaluate.main([
        '--convert-model',
        'csv',
        'tests/test-data/csv/annotations.csv',
        'tests/test-data/csv/classes.csv',
        'tests/snapshot/resnet50_csv_01.h5'
    ])
