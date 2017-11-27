"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

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

import keras_retinanet.losses
import keras_retinanet.layers
from keras_retinanet.models.resnet import ResNet50RetinaNet
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.utils.keras_version import check_keras_version

import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(weights='imagenet', multi_gpu=0):
    # create "base" model (no NMS)
    image = keras.layers.Input((None, None, 3))
    model = ResNet50RetinaNet(image, num_classes=20, weights=weights, nms=False)

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
    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join('snapshots', 'resnet50_voc_{epoch:02d}.h5'), verbose=1)
    checkpoint.set_model(prediction_model)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

    return [checkpoint, lr_scheduler]


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for Pascal VOC object detection.')
    parser.add_argument('voc_path', help='Path to Pascal VOC directory (ie. /tmp/VOCdevkit/VOC2007).')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).', default='imagenet')
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

    # create the model
    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(weights=args.weights, multi_gpu=args.multi_gpu)

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

    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for training data
    train_generator = PascalVocGenerator(
        args.voc_path,
        'trainval',
        train_image_data_generator,
        batch_size=args.batch_size
    )

    # create a generator for testing data
    val_generator = PascalVocGenerator(
        args.voc_path,
        'test',
        val_image_data_generator,
        batch_size=args.batch_size
    )

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=10000,
        epochs=50,
        verbose=1,
        validation_data=val_generator,
        validation_steps=3000,  # len(val_generator.image_names) // args.batch_size,
        callbacks=create_callbacks(model, training_model, prediction_model),
    )

    # store final result too
    model.save(os.path.join('snapshots', 'resnet50_voc_final.h5'))
