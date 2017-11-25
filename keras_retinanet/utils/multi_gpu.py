from __future__ import print_function
import keras
from keras.models import *
from keras.layers import Input, merge, Lambda
from keras.layers.merge import Concatenate
from keras.layers.merge import concatenate
from keras import backend as K
from keras.models import Model

import tensorflow as tf

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
session = tf.Session(config=session_config)


def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]


def to_multi_gpu(model, n_gpus=2):
    if n_gpus == 1:
        return model

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:])
    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        # Deprecated
        # merged = merge(towers, mode='concat', concat_axis=0)
        # merged = concatenate(inputs=towers, axis=0)
        merged = Concatenate(axis=0)(towers)
    return Model(inputs=[x], outputs=merged)


def make_parallel(model, gpu_list):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    gpu_count = len(gpu_list)
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % gpu_list[i]):
            with tf.name_scope('tower_%d' % gpu_list[i]) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for idx, outputs in enumerate(outputs_all):
            if idx == 1:
                merged.append(concatenate(outputs, axis=0, name='regression'))
            elif idx == 2:
                merged.append(concatenate(outputs, axis=0, name='classification'))
            else:
                merged.append(concatenate(outputs, axis=0))

        return Model(inputs=model.inputs, outputs=merged)
