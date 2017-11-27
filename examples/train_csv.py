"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os

import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model

import tensorflow as tf

import keras_retinanet.layers
from keras_retinanet.models.resnet import ResNet50RetinaNet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
import keras_retinanet


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(num_classes, weights='imagenet', multi_gpu=0):
    # create "base" model (no NMS)
    image = keras.layers.Input((None, None, 3))
    model = ResNet50RetinaNet(image, num_classes=num_classes, weights=weights, nms=False)

    # optionally wrap in a parallel model
    if args.multi_gpu > 1:
        training_model = multi_gpu_model(model, gpus=args.multi_gpu)
    else:
        training_model = model

    # append NMS for prediction
    detections = keras_retinanet.layers.NonMaximumSuppression(name='nms')(model.outputs)
    prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model):
    # save the prediction model
    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join('snapshots', 'resnet50_csv_{epoch:02d}.h5'), verbose=1)
    checkpoint.set_model(prediction_model)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    return [checkpoint, lr_scheduler]


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for object detection from a CSV file.')
    parser.add_argument('train_path', help='Path to CSV file for training (required)')
    parser.add_argument('classes', help='Path to a CSV file containing class label mapping (required)')
    parser.add_argument('--val_path', help='Path to CSV file for validation (optional')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).',
                        default='imagenet')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )

    # create a generator for training data
    train_generator = CSVGenerator(
        csv_data_file=args.train_path,
        csv_class_file=args.classes,
        image_data_generator=train_image_data_generator,
        batch_size=args.batch_size
    )

    if args.val_path:
        test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

        # create a generator for testing data
        test_generator = CSVGenerator(
            csv_data_file=args.val_path,
            csv_class_file=args.classes,
            image_data_generator=test_image_data_generator,
            batch_size=args.batch_size
        )
    else:
        test_generator = None

    num_classes = train_generator.num_classes()

    # create the model
    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(
        num_classes=num_classes,
        weights=args.weights,
        multi_gpu=args.multi_gpu
    )

    # compile model (note: set loss to None since loss is added inside layer)
    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    # print model summary
    print(model.summary())

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.size() // args.batch_size,
        epochs=20,
        verbose=1,
        max_queue_size=20,
        validation_data=test_generator,
        validation_steps=test_generator.size() // args.batch_size if test_generator else 0,
        callbacks=create_callbacks(model, training_model, prediction_model),
    )

    # store final result too
    model.save(os.path.join('snapshots', 'resnet50_csv_final.h5'))
