import keras.applications.imagenet_utils
import keras.preprocessing.image
import keras.backend

import cv2
import xml.etree.ElementTree as ET

import os
import numpy as np
import time

voc_classes = {
	'__background__' : 0,
	'aeroplane'      : 1,
	'bicycle'        : 2,
	'bird'           : 3,
	'boat'           : 4,
	'bottle'         : 5,
	'bus'            : 6,
	'car'            : 7,
	'cat'            : 8,
	'chair'          : 9,
	'cow'            : 10,
	'diningtable'    : 11,
	'dog'            : 12,
	'horse'          : 13,
	'motorbike'      : 14,
	'person'         : 15,
	'pottedplant'    : 16,
	'sheep'          : 17,
	'sofa'           : 18,
	'train'          : 19,
	'tvmonitor'      : 20
}

class PascalVocIterator(keras.preprocessing.image.Iterator):
	def __init__(
		self,
		data_dir,
		set_name,
		image_data_generator,
		classes=voc_classes,
		image_shape=(512, 512, 3),
		image_extension='.jpg',
		skip_truncated=False,
		skip_difficult=False,
		batch_size=1,
		shuffle=True,
		seed=np.uint32(time.time() * 1000),
	):
		self.data_dir             = data_dir
		self.set_name             = set_name
		self.classes              = classes
		self.image_names          = [l.strip() for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
		self.image_data_generator = image_data_generator
		self.image_shape          = image_shape # TODO: Make keras-retinanet accept any input shape
		self.image_scale          = 1.0
		self.image_extension      = image_extension
		self.skip_truncated       = skip_truncated
		self.skip_difficult       = skip_difficult

		super().__init__(len(self.image_names), batch_size, shuffle, seed)

	def parse_annotations(self, filename):
		boxes = np.zeros((0, 5), dtype=keras.backend.floatx())

		tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename + '.xml'))
		root = tree.getroot()

		width = float(root.find('size').find('width').text)
		height = float(root.find('size').find('height').text)

		for o in root.iter('object'):
			if int(o.find('truncated').text) and self.skip_truncated:
				continue

			if int(o.find('difficult').text) and self.skip_difficult:
				continue

			box = np.zeros((1, 5), dtype=keras.backend.floatx())

			class_name = o.find('name').text
			if class_name not in self.classes:
				raise Exception('Class name "{}" not found in classes "{}"'.format(class_name, self.classes))
			box[0, 4] = self.classes[class_name]

			bndbox = o.find('bndbox')
			box[0, 0] = (float(bndbox.find('xmin').text) - 1) / width * self.image_shape[1]
			box[0, 1] = (float(bndbox.find('ymin').text) - 1) / height * self.image_shape[0]
			box[0, 2] = (float(bndbox.find('xmax').text) - 1) / width * self.image_shape[1]
			box[0, 3] = (float(bndbox.find('ymax').text) - 1) / height * self.image_shape[0]

			boxes = np.append(boxes, box, axis=0)

		return boxes

	def next(self):
		# Lock indexing to prevent race conditions
		with self.lock:
			selection, _, batch_size = next(self.index_generator)

		# Transformation of images is not under thread lock so it can be done in parallel
		image_batch      = np.zeros((batch_size,) + self.image_shape, dtype=keras.backend.floatx())
		gt_boxes_batch   = np.zeros((batch_size, 0, 5), dtype=keras.backend.floatx())

		for batch_index, image_index in enumerate(selection):
			path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
			image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
			image = cv2.resize(image, self.image_shape[:2]).astype(keras.backend.floatx())
			#image = self.image_data_generator.random_transform(image)

			# Copy image to batch blob
			image_batch[batch_index] = image

			# Set ground truth boxes
			boxes = np.expand_dims(self.parse_annotations(self.image_names[image_index]), axis=0)
			gt_boxes_batch = np.append(gt_boxes_batch, boxes, axis=1)

			# Scale the ground truth boxes to the selected image scale
			gt_boxes_batch[batch_index, :, :4] *= self.image_scale

		# Convert the image to zero-mean
		image_batch = keras.applications.imagenet_utils.preprocess_input(image_batch)
		image_batch = self.image_data_generator.standardize(image_batch)

		return [image_batch, gt_boxes_batch], None


class ObjectDetectionGenerator:
	def flow(self, data, classes, image_shape):
		return PascalVocIterator(data, classes, keras.preprocessing.image.ImageDataGenerator(), image_shape)
