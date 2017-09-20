import sys
sys.path.append('.')

import argparse

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import keras
import keras.preprocessing.image

from keras_retinanet.models import ResNet50RetinaNet
from keras_retinanet.preprocessing import PascalVocIterator

def create_model():
	image = keras.layers.Input((None, None, 3))
	gt_boxes = keras.layers.Input((None, 5))
	return ResNet50RetinaNet([image, gt_boxes], num_classes=21)

def parse_args():
	parser = argparse.ArgumentParser(description='Simple training script for Pascal VOC object detection.')
	parser.add_argument('voc_path', help='Path to Pascal VOC directory (ie. /tmp/VOCdevkit/VOC2007).')

	return parser.parse_args()


if __name__=='__main__':
	# parse arguments
	args = parse_args()

	# create the model
	print('Creating model, this may take a second...')
	model = create_model()

	# compile model (note: set loss to None since loss is added inside layer)
	model.compile(loss=None, optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

	# print model summary
	print(model.summary())

	# create image data generator objects
	train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
		rescale=1.0/255.0,
		horizontal_flip=True,
		#vertical_flip=True,
		width_shift_range=0.1,
		height_shift_range=0.1,
		zoom_range=0.1,
	)
	test_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
		rescale=1.0/255.0,
	)

	# create a generator for training data
	train_generator = PascalVocIterator(
		args.voc_path,
		'trainval',
		train_image_data_generator
	)

	# create a generator for testing data
	test_generator = PascalVocIterator(
		args.voc_path,
		'test',
		test_image_data_generator
	)

	# start training
	batch_size = 1
	model.fit_generator(
		generator=train_generator,
		steps_per_epoch=len(train_generator.image_names) // batch_size,
		epochs=100,
		verbose=1,
		validation_data=test_generator,
		validation_steps=500, #len(test_generator.image_names) // batch_size,
		callbacks=[
			keras.callbacks.ModelCheckpoint('snapshots/resnet50_voc_best.h5', monitor='val_loss', verbose=1, save_best_only=True),
			keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
		],
	)

	# store final result too
	model.save('snapshots/resnet50_voc_final.h5')
