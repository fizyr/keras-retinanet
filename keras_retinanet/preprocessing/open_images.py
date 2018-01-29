import csv
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from .generator import Generator
from ..utils.image import read_image_bgr


def get_labels(metadata_dir):
    trainable_classes_path = os.path.join(metadata_dir, 'classes-bbox-trainable.txt')
    description_path = os.path.join(metadata_dir, 'class-descriptions.csv')

    description_table = {}
    with open(description_path) as f:
        for row in csv.reader(f):
            if len(row):
                description_table[row[0]] = row[1].replace("\"", "").replace("'", "").replace('`', '')

    with open(trainable_classes_path, 'rb') as f:
        trainable_classes = f.read().split('\n')

    id2labels_dict = dict([(i, description_table[c]) for i, c in enumerate(trainable_classes)])
    cls_index_dict = dict([(c, i) for i, c in enumerate(trainable_classes)])

    return id2labels_dict, cls_index_dict


def generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index_dict):
    annotations_path = os.path.join(metadata_dir, subset, 'annotations-human-bbox.csv')

    cnt = 0
    with open(annotations_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=['ImageID', 'Source', 'LabelName',
                                            'Confidence', 'XMin', 'XMax', 'YMin',
                                            'YMax'])
        reader.next()
        for _ in reader:
            cnt += 1

        print ('There are {} lines in subset: {}'.format(cnt, subset))

    id_annotations = dict()

    with open(annotations_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=['ImageID', 'Source', 'LabelName',
                                            'Confidence', 'XMin', 'XMax', 'YMin',
                                            'YMax'])
        reader.next()

        images_sizes = dict()
        for row in tqdm(reader, total=cnt):
            frame = row['ImageID']

            class_name = row['LabelName']

            if class_name not in cls_index_dict:
                continue

            cls_id = cls_index_dict[class_name]

            img_path = os.path.join(main_dir, 'images', subset, frame + '.jpg')
            if frame in images_sizes:
                width, height = images_sizes[frame]
            else:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.width, img.height
                        images_sizes[frame] = (width, height)
                except Exception:
                    continue

            x1 = float(row['XMin'])
            x2 = float(row['XMax'])
            y1 = float(row['YMin'])
            y2 = float(row['YMax'])

            x1_abs = int(round(x1 * width))
            x2_abs = int(round(x2 * width))
            y1_abs = int(round(y1 * height))
            y2_abs = int(round(y2 * height))

            if y2_abs <= y1_abs:
                print ('{} w: {} h: {}'.format(row, width, height))
                continue

            if x2_abs <= x1_abs:
                print ('{} w: {} h: {}'.format(row, width, height))
                continue

            img_id = row['ImageID']

            annotation = {'cls_id': cls_id, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}

            if img_id in id_annotations:

                annotations = id_annotations[img_id]
                annotations['boxes'].append(annotation)
            else:
                id_annotations[img_id] = {'w': width, 'h': height, 'boxes': [annotation]}
    return id_annotations


class OpenImagesGenerator(Generator):
    def __init__(
            self, main_dir, subset, labels_filter,
            **kwargs
    ):
        metadata_dir = os.path.join(main_dir, '2017_11')
        fname_json = os.path.join(metadata_dir, subset, subset + '.json')
        self.base_dir = os.path.join(main_dir, 'images', subset)

        labels_dict = None if labels_filter is None else dict([(l, i) for i, l in enumerate(labels_filter)])

        print ('loading {} subset'.format(subset))
        self.id2labels_dict, cls_index_dict = get_labels(metadata_dir)

        if os.path.exists(fname_json):
            with open(fname_json, 'r') as f:
                self.annotations = json.loads(f.read())
        else:
            self.annotations = generate_images_annotations_json(main_dir, metadata_dir, subset, cls_index_dict)
            json.dump(self.annotations, open(fname_json, "w"))

        if labels_dict is not None:
            filtered_annotations = dict()

            for k in self.annotations:
                img_ann = self.annotations[k]

                filtered_boxes = []
                for ann in img_ann['boxes']:
                    cls_id = ann['cls_id']
                    label = self.id2labels_dict[cls_id]
                    if label in labels_dict:
                        ann['cls_id'] = labels_dict[label]
                        filtered_boxes.append(ann)

                if len(filtered_boxes) > 0:
                    filtered_annotations[k] = {'w': img_ann['w'], 'h': img_ann['h'], 'boxes': filtered_boxes}

            self.id2labels_dict = dict([(labels_dict[k], k) for k in labels_dict])
            self.annotations = filtered_annotations

        self.id2imageid = dict()
        for i, k in enumerate(self.annotations):
            self.id2imageid[i] = k

        super(OpenImagesGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.annotations)

    def num_classes(self):
        return len(self.id2labels_dict)

    def name_to_label(self, name):
        raise NotImplementedError()

    def label_to_name(self, label):
        return self.id2labels_dict[label]

    def image_aspect_ratio(self, image_index):
        img_annotations = self.annotations[self.id2imageid[image_index]]
        height, width = img_annotations['h'], img_annotations['w']
        return float(width) / float(height)

    def image_path(self, image_index):
        path = os.path.join(self.base_dir, self.id2imageid[image_index] + '.jpg')
        return path

    def load_image(self, image_index):
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        image_annotations = self.annotations[self.id2imageid[image_index]]

        labels = image_annotations['boxes']
        height, width = image_annotations['h'], image_annotations['w']

        boxes = np.zeros((len(labels), 5))

        for idx, ann in enumerate(labels):
            cls_id = ann['cls_id']
            x1 = ann['x1'] * width
            x2 = ann['x2'] * width
            y1 = ann['y1'] * height
            y2 = ann['y2'] * height

            boxes[idx, 0] = x1
            boxes[idx, 1] = y1
            boxes[idx, 2] = x2
            boxes[idx, 3] = y2
            boxes[idx, 4] = cls_id

        return boxes
