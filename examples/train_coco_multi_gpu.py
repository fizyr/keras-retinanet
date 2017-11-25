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

import tensorflow as tf

import keras_retinanet.callbacks.coco
import keras_retinanet.losses
from keras_retinanet.models.resnet import ResNet50RetinaNetMultiGpu
from keras_retinanet.preprocessing.coco import CocoGenerator
from keras_retinanet.utils.keras_version import check_keras_version


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_model(gpu_list, weights='imagenet'):
    image = keras.layers.Input((None, None, 3))
    return ResNet50RetinaNetMultiGpu(image, gpu_list=gpu_list, num_classes=80, weights=weights)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for COCO object detection.')
    parser.add_argument('coco_path', help='Path to COCO directory (ie. /tmp/COCO).')
    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).',
                        default='imagenet')
    parser.add_argument('--batch-size', help='Size of the batches.', default=2, type=int)
    parser.add_argument('--gpu-list', help='List of the GPUs to use (as reported by nvidia-smi). Sample input: 0,1',
                        default='0,1')

    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    gpu_list = map(int, args.gpu_list.split(','))
    keras.backend.tensorflow_backend.set_session(get_session())

    # Batches should be divided into equal subsets for each gpu
    assert args.batch_size % len(gpu_list) is 0

    # create the model
    print('Creating model, this may take a second...')
    model = create_model(gpu_list=gpu_list, weights=args.weights)

    # compile model (note: set loss to None since loss is added inside layer)
    model.compile(
        loss={
            'regression': keras_retinanet.losses.smooth_l1(),
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
    train_generator = CocoGenerator(
        args.coco_path,
        'train2017',
        train_image_data_generator,
        batch_size=args.batch_size
    )

    # create a generator for testing data
    val_generator = CocoGenerator(
        args.coco_path,
        'val2017',
        val_image_data_generator,
        batch_size=args.batch_size
    )

    # Steps per epoch should be divisible by batch size to
    steps_per_epoch = int(args.batch_size * (20000 // args.batch_size))

    # start training
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join('snapshots', 'resnet50_coco_best.h5'), monitor='loss',
                                            verbose=1, save_best_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto',
                                              epsilon=0.0001, cooldown=0, min_lr=0),
            keras_retinanet.callbacks.coco.CocoEvalMultiGpu(val_generator,
                                                            model_path=os.path.join('snapshots',
                                                                                    'resnet50_coco_best.h5')),
        ],
    )

    # store final result too
    model.save(os.path.join('snapshots', 'resnet50_coco_final.h5'))
