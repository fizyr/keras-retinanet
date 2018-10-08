import json
import logging
import pickle
from copy import deepcopy

import os.path

import numpy as np

from .generator import Generator
from ..utils.image import read_image_bgr


# {
#     "path": string,
#     "count_to_process": "all" or "num",
#     "only_verified": false or true,
#     "max_bbox_area": num (in relative coordinates, 0.0 - 1.0),
#     "min_bbox_area": num (in relative coordinates, 0.0 - 1.0)
# }
def _load_one_path(images_dir, path, count_to_process, only_verified, min_bbox_area, max_bbox_area):
    train_data = {
        'images_with_annotations': [],
        'categories': []
    }

    current_path = os.path.join(images_dir, path)
    if not os.path.isfile(os.path.join(current_path, 'annotations.pickle')) and \
            not os.path.isfile(os.path.join(current_path, 'annotations.json')):
        logging.warning("Error path: {}".format(os.path.join(current_path, 'annotations.pickle')))
        return train_data

    if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
        with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
            annotations = pickle.load(f)
    elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
        with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
            annotations = json.load(f)

    assert annotations,\
        "Some error in annotation loading: {}".format(os.path.join(current_path, 'annotations'))

    if only_verified:
        annotations['images'] = [image for image in annotations['images']
                                 if any(map(lambda x: x, image['verified'].values()))]

    for image in annotations['images']:
        image['file_name'] = os.path.join(images_dir, path, image['file_name'])

    if count_to_process != "all":
        np.random.shuffle(annotations['images'])
        annotations['images'] = annotations['images'][:int(count_to_process)]

    if len(train_data['categories']) == 0:
        train_data['categories'] = annotations['categories']
    else:
        current_categories = set(map(lambda x: (x['id'], x['name']), annotations['categories']))
        for category in train_data['categories']:
            cat = (category['id'], category['name'])
            assert cat in current_categories, 'Categories ids must be same in all datasets'

    image_id_to_image = {image['id']: image for image in annotations['images']}
    images_with_annotations = {image['id']: [] for image in annotations['images']}
    for annotation in annotations['annotations']:
        if annotation['image_id'] not in image_id_to_image:
            continue
        image = image_id_to_image[annotation['image_id']]
        image_area = image['width'] * image['height']
        bbox_area = (annotation['bbox'][1][0] * image['width'] - annotation['bbox'][0][0] * image['width']) * \
                    (annotation['bbox'][1][1] * image['height'] - annotation['bbox'][0][1] * image['height'])
        area_ratio = bbox_area / image_area

        if area_ratio < min_bbox_area or area_ratio > max_bbox_area:
            continue

        annotation['category_id'] += 1  # Background category = 0
        images_with_annotations[annotation['image_id']].append(annotation)

    for image_id, anns in images_with_annotations.items():
        image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
        train_data['images_with_annotations'].append((image, anns))

    return train_data


def _load_images(path, subset='train'):
    assert os.path.isfile(path), 'There is no annotations'

    with open(path, 'r') as f:
        config = json.load(f)

    images_dir = config['train']['images_dir']

    if subset == 'train':
        datasets = config['train']['datasets_to_train']
    elif subset == 'val':
        datasets = config['train']['datasets_to_validate']
    else:
        assert False, "Unknown subset: {}".format(subset)

    data_annotations = {
        'images_with_annotations': [],
        'categories': []
    }
    # Train dataset loading
    for dataset in datasets:
        data = _load_one_path(images_dir, dataset['path'], dataset['count_to_process'], dataset['only_verified'],
                              dataset['min_bbox_area'], dataset['max_bbox_area'])
        data_annotations['images_with_annotations'].extend(data['images_with_annotations'])
        data_annotations['categories'].extend(data['categories'])

    return {i: image_ann for i, image_ann in enumerate(data_annotations['images_with_annotations'])}, \
           {category['id'] + 1: category['name'] for category in data_annotations['categories']}


class TrassirGenerator(Generator):
    def __init__(
        self,
        annotations_path,
        subset='train',
        labels=['person'],
        **kwargs
    ):
        self.images, self.categories = _load_images(annotations_path, subset)
        self.categories[0] = 'background'

        # Filtering categories and images annotations
        self.images = {
            i: (image_annotations[0], list(filter(lambda x: self.categories[x['category_id']] in labels,
                                                  image_annotations[1])))
            for i, image_annotations in self.images.items()
        }

        labels.append('background')
        self.categories = {i: cat for i, cat in self.categories.items() if cat in labels}
        self.category_name_to_id = dict(zip(self.categories.values(), self.categories.keys()))
        
        super(TrassirGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.images)

    def num_classes(self):
        return len(self.categories)

    def name_to_label(self, name):
        return self.category_name_to_id[name]

    def label_to_name(self, label):
        # if label == 0:
        #     return 'BG'
        return self.categories[label]

    def image_aspect_ratio(self, image_index):
        image, _ = self.images[image_index]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        return read_image_bgr(self.images[image_index][0]['file_name'])

    def load_annotations(self, image_index):
        image, annotations = self.images[image_index]

        boxes = np.zeros((len(annotations), 5))
        for idx, ann in enumerate(annotations):
            boxes[idx, 0] = ann['bbox'][0][0] * image['width']
            boxes[idx, 1] = ann['bbox'][0][1] * image['height']
            boxes[idx, 2] = ann['bbox'][1][0] * image['width']
            boxes[idx, 3] = ann['bbox'][1][1] * image['height']
            boxes[idx, 4] = ann['category_id']

            boxes[idx, 0] = boxes[idx, 0] if boxes[idx, 0] > 0. else 0.
            boxes[idx, 1] = boxes[idx, 1] if boxes[idx, 1] > 0. else 0.
            boxes[idx, 2] = boxes[idx, 2] if boxes[idx, 2] < image['width'] else image['width']
            boxes[idx, 3] = boxes[idx, 3] if boxes[idx, 3] < image['height'] else image['height']

        return boxes
