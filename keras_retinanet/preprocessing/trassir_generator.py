import json
import logging
import pickle

import os.path

import numpy as np
from PIL import Image

from .generator import Generator
from ..utils.image import read_image_bgr

kitti_classes = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 7
}

ANNOTATIONS_PATH = 'trassir_annotations.json'

# {
#     "path": string,
#     "count_to_process": "all" or "num",
#     "only_verified": false or true,
#     "max_bbox_area": num (in relative coordinates, 0.0 - 1.0),
#     "min_bbox_area": num (in relative coordinates, 0.0 - 1.0),
#     "keep_without_annotations": false or true
# }
def _load_one_path(images_dir, path, count_to_process, only_verified, max_bbox_area, min_bbox_area):
    current_path = os.path.join(images_dir, path, keep_without_annotations)
    if not os.path.isfile(os.path.join(current_path, 'annotations.pickle')) and \
            not os.path.isfile(os.path.join(current_path, 'annotations.json')):
        logging.warning("Error path: {}".format(os.path.join(current_path, 'annotations.pickle')))

    if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
        with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
            annotations = pickle.load(f)
    elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
        with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
            annotations = json.load(f)

    assert annotations,\
        "Some error in annotation loading: {}".format(os.path.join(current_path, 'annotations.pickle'))

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
        assert sorted(list(map(lambda x: (x['id'], x['name']), annotations['categories']))) == \
               sorted(list(map(lambda x: (x['id'], x['name']), train_data['categories']))), \
            'Categories must be same in all datasets'

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

        if area_ratio < dataset['min_bbox_area'] or area_ratio > dataset['max_bbox_area']:
            continue

        images_with_annotations[annotation['image_id']].append(annotation)

    images_with_annotations = {image_id: anns for image_id, anns in images_with_annotations.items()
                               if len(anns) > 0}
    for image_id, anns in images_with_annotations.items():
        image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
        image['id'] = train_last_image_index
        for annotation in anns:
            annotation['image_id'] = train_last_image_index
        train_data['images_with_annotations'].append((image, anns))
        train_last_image_index += 1


def _load_images():
    assert os.path.isfile(ANNOTATIONS_PATH), 'There is no annotations'

    with open(ANNOTATIONS_PATH, 'r') as f:
        images_dir = json.load(f)

    train_last_image_index = 0
    train_data = {
        'images_with_annotations': [],
        'categories': []
    }

    # Train dataset loading
    for dataset in config['train']['datasets_to_train']:
        current_path = os.path.join(images_dir, dataset['path'])
        assert os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or \
            os.path.isfile(os.path.join(current_path, 'annotations.json')), \
            "Error path: {}".format(os.path.join(current_path, 'annotations.pickle'))

        if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = pickle.load(f)
        elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = json.load(f)

        assert annotations

        if dataset['only_verified']:
            annotations['images'] = [image for image in annotations['images']
                                     if any(map(lambda x: x, image['verified'].values()))]

        for image in annotations['images']:
            image['file_name'] = os.path.join(images_dir, dataset['path'], image['file_name'])

        if dataset['count_to_process'] != "all":
            np.random.shuffle(annotations['images'])
            annotations['images'] = annotations['images'][:int(dataset['count_to_process'])]

        if len(train_data['categories']) == 0:
            train_data['categories'] = annotations['categories']
        else:
            assert sorted(list(map(lambda x: (x['id'], x['name']), annotations['categories']))) == \
                   sorted(list(map(lambda x: (x['id'], x['name']), train_data['categories']))), \
                'Categories must be same in all datasets'

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

            if area_ratio < dataset['min_bbox_area'] or area_ratio > dataset['max_bbox_area']:
                continue

            images_with_annotations[annotation['image_id']].append(annotation)

        images_with_annotations = {image_id: anns for image_id, anns in images_with_annotations.items()
                                   if len(anns) > 0}
        for image_id, anns in images_with_annotations.items():
            image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
            image['id'] = train_last_image_index
            for annotation in anns:
                annotation['image_id'] = train_last_image_index
            train_data['images_with_annotations'].append((image, anns))
            train_last_image_index += 1


    val_last_image_index = 0
    validation_data = {
        'images_with_annotations': [],
        'categories': []
    }

    # Validation dataset loading
    for dataset in config['train']['datasets_to_validate']:
        current_path = os.path.join(images_dir, dataset['path'])
        assert os.path.isfile(os.path.join(current_path, 'annotations.pickle')) or \
            os.path.isfile(os.path.join(current_path, 'annotations.json'))

        if os.path.isfile(os.path.join(current_path, 'annotations.pickle')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = pickle.load(f)
        elif os.path.isfile(os.path.join(current_path, 'annotations.json')):
            with open(os.path.join(current_path, 'annotations.pickle'), 'rb') as f:
                annotations = json.load(f)

        assert annotations

        if dataset['only_verified']:
            annotations['images'] = [image for image in annotations['images']
                                     if any(map(lambda x: x, image['verified'].values()))]

        for image in annotations['images']:
            image['file_name'] = os.path.join(images_dir, dataset['path'], image['file_name'])

        if dataset['count_to_process'] != "all":
            np.random.shuffle(annotations['images'])
            annotations['images'] = annotations['images'][:int(dataset['count_to_process'])]

        if len(validation_data['categories']) == 0:
            validation_data['categories'] = annotations['categories']
        else:
            assert map(lambda x: (x['id'], x['name']), annotations['categories']) == \
                   map(lambda x: (x['id'], x['name']), validation_data['categories']), \
                'Categories must be same in all datasets'

        image_id_to_image = {image['id']: image for image in annotations['images']}
        images_with_annotations = {image['id']: [] for image in annotations['images']}
        for annotation in annotations['annotations']:
            if annotation['image_id'] not in image_id_to_image:
                continue
            image = image_id_to_image[annotation['image_id']]
            image_area = image['width'] * image['height']
            bbox_area = (annotation['bbox'][1][0] * image['width'] - annotation['bbox'][0][0] * image['width']) * \
                        (annotation['bbox'][1][1] * image['height'] - annotation['bbox'][0][1] * image['height'])

            if dataset['min_bbox_area'] > bbox_area / image_area > dataset['max_bbox_area']:
                continue

            images_with_annotations[annotation['image_id']].append(annotation)

        images_with_annotations = {image_id: anns for image_id, anns in images_with_annotations.items()
                                   if len(anns) > 0}
        for image_id, anns in images_with_annotations.items():
            image, anns = deepcopy(image_id_to_image[image_id]), deepcopy(anns)
            image['id'] = val_last_image_index
            for annotation in anns:
                annotation['image_id'] = val_last_image_index
            validation_data['images_with_annotations'].append((image, anns))
            val_last_image_index += 1

    return train_data, validation_data


class KittiGenerator(Generator):
    def __init__(
        self,
        base_dir,
        subset='train',
        **kwargs
    ):
        self.base_dir = base_dir

        label_dir = os.path.join(self.base_dir, subset, 'labels')
        image_dir = os.path.join(self.base_dir, subset, 'images')

        """
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        """

        self.id_to_labels = {}
        for label, id in kitti_classes.items():
            self.id_to_labels[id] = label

        self.image_data = dict()
        self.images = []
        for i, fn in enumerate(os.listdir(label_dir)):
            label_fp = os.path.join(label_dir, fn)
            image_fp = os.path.join(image_dir, fn.replace('.txt', '.png'))

            self.images.append(image_fp)

            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'left', 'top', 'right', 'bottom', 'dh', 'dw', 'dl',
                          'lx', 'ly', 'lz', 'ry']
            with open(label_fp, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
                boxes = []
                for line, row in enumerate(reader):
                    label = row['type']
                    cls_id = kitti_classes[label]

                    annotation = {'cls_id': cls_id, 'x1': row['left'], 'x2': row['right'], 'y2': row['bottom'], 'y1': row['top']}
                    boxes.append(annotation)

                self.image_data[i] = boxes

        super(KittiGenerator, self).__init__(**kwargs)

    def size(self):
        return len(self.images)

    def num_classes(self):
        return max(kitti_classes.values()) + 1

    def name_to_label(self, name):
        raise NotImplementedError()

    def label_to_name(self, label):
        return self.id_to_labels[label]

    def image_aspect_ratio(self, image_index):
        # PIL is fast for metadata
        image = Image.open(self.images[image_index])
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        return read_image_bgr(self.images[image_index])

    def load_annotations(self, image_index):
        annotations = self.image_data[image_index]

        boxes = np.zeros((len(annotations), 5))
        for idx, ann in enumerate(annotations):
            boxes[idx, 0] = float(ann['x1'])
            boxes[idx, 1] = float(ann['y1'])
            boxes[idx, 2] = float(ann['x2'])
            boxes[idx, 3] = float(ann['y2'])
            boxes[idx, 4] = int(ann['cls_id'])
        return boxes
