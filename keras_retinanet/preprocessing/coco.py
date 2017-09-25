from __future__ import division

import keras.applications.imagenet_utils
import keras.preprocessing.image
import keras.backend

from .image import random_transform_batch, resize_image

import cv2

import os
import numpy as np
import time

from pycocotools.coco import COCO


class CocoIterator(keras.preprocessing.image.Iterator):
    def __init__(
        self,
        data_dir,
        set_name,
        image_data_generator,
        image_min_side=600,
        image_max_side=1024,
        batch_size=1,
        shuffle=True,
        seed=None,
    ):
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.coco                 = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        self.image_ids            = self.coco.getImgIds()
        self.image_data_generator = image_data_generator
        self.image_min_side       = image_min_side
        self.image_max_side       = image_max_side

        if seed is None:
            seed = np.uint32(time.time() * 1000)

        self.load_classes()

        assert(batch_size == 1), "Currently only batch_size=1 is allowed."

        super(CocoIterator, self).__init__(len(self.image_ids), batch_size, shuffle, seed)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.classes = {'__background__': 0}
        for c in categories:
            self.classes[c['name']] = c['id']

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def next(self):
        # lock indexing to prevent race conditions
        with self.lock:
            selection, _, batch_size = next(self.index_generator)

        assert(batch_size == 1), "Currently only batch_size=1 is allowed."

        boxes_batch = np.zeros((batch_size, 0, 5), dtype=keras.backend.floatx())

        for batch_index, image_index in enumerate(selection):
            coco_image = self.coco.loadImgs(self.image_ids[image_index])[0]
            path  = os.path.join(self.data_dir, self.set_name, coco_image['file_name'])
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image, image_scale = resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

            # copy image to image batch (currently only batch_size==1 is allowed)
            image_batch = np.expand_dims(image, axis=0).astype(keras.backend.floatx())

            # set ground truth boxes
            annotations_ids = self.coco.getAnnIds(imgIds=coco_image['id'], iscrowd=False)

            # some images appear to miss annotations (like image with id 257034)
            if len(annotations_ids) == 0:
                return self.next()

            # parse annotations
            annotations = self.coco.loadAnns(annotations_ids)
            for idx, a in enumerate(annotations):
                box = np.zeros((1, 1, 5), dtype=keras.backend.floatx())
                box[0, 0, :4] = a['bbox']
                box[0, 0, 4]  = a['category_id']
                boxes_batch = np.append(boxes_batch, box, axis=1)

            # transform from [x, y, w, h] to [x1, y1, x2, y2]
            boxes_batch[batch_index, :, 2] = boxes_batch[batch_index, :, 0] + boxes_batch[batch_index, :, 2]
            boxes_batch[batch_index, :, 3] = boxes_batch[batch_index, :, 1] + boxes_batch[batch_index, :, 3]

            # scale the ground truth boxes to the selected image scale
            boxes_batch[batch_index, :, :4] *= image_scale

        # randomly transform images and boxes simultaneously
        image_batch, boxes_batch = random_transform_batch(image_batch, boxes_batch, self.image_data_generator)

        # convert the image to zero-mean
        image_batch = keras.applications.imagenet_utils.preprocess_input(image_batch)
        image_batch = self.image_data_generator.standardize(image_batch)

        return [image_batch, boxes_batch], None
