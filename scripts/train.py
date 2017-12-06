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

import tensorflow as tf

import keras_retinanet.losses
import keras_retinanet.layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.models.resnet import ResNet50RetinaNet
from keras_retinanet.utils.keras_version import check_keras_version


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(num_classes, weights='imagenet', multi_gpu=0):
    # create "base" model (no NMS)
    image = keras.layers.Input((None, None, 3))

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if args.multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = ResNet50RetinaNet(image, num_classes=num_classes, weights=weights, nms=False)
        training_model = multi_gpu_model(model, gpus=args.multi_gpu)
    else:
        model = ResNet50RetinaNet(image, num_classes=num_classes, weights=weights, nms=False)
        training_model = model

    # append NMS for prediction
    detections = keras_retinanet.layers.NonMaximumSuppression(name='nms')(model.outputs)
    prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])

    # compile model
    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, dataset_type):
    callbacks = []

    # save the prediction model
    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join('snapshots', 'resnet50_{dataset_type}_{{epoch:02d}}.h5'.format(dataset_type=dataset_type)), verbose=1)
    checkpoint = RedirectModel(checkpoint, prediction_model)
    callbacks.append(checkpoint)

    if dataset_type == 'coco':
        import keras_retinanet.callbacks.coco

        # use prediction model for evaluation
        evaluation = keras_retinanet.callbacks.coco.CocoEval(validation_generator)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    callbacks.append(lr_scheduler)

    return callbacks


def create_generators(args):
    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from keras_retinanet.preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            train_image_data_generator,
            batch_size=args.batch_size
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            val_image_data_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            train_image_data_generator,
            batch_size=args.batch_size
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            val_image_data_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_datrain_image_data_generator,
            batch_size=args.batch_size
        )

        if args.val_path:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                val_image_data_generator,
                batch_size=args.batch_size
            )
        else:
            validation_generator = None
    else:
        raise ValueError('Invalid data type received: {}'.format(dataset_type))

    return train_generator, validation_generator


def parse_args():
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')

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

    # create the generators
    train_generator, validation_generator = create_generators(args)

    # create the model
    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(num_classes=train_generator.num_classes(), weights=args.weights, multi_gpu=args.multi_gpu)

    # print model summary
    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(model, training_model, prediction_model, validation_generator, args.dataset_type)

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=10000,
        epochs=50,
        verbose=1,
        callbacks=callbacks,
    )
