import argparse

import h5py
import numpy as np
from tqdm import tqdm

from ..preprocessing.csv_generator import CSVGenerator
from ..models import backbone


def parse():
    parser = argparse.ArgumentParser(description='Simple script for building an HDF5 file for retinanet training.')

    parser.add_argument('--train-annotations',
                        help='Path to CSV file containing annotations for training.',
                        required=True)
    parser.add_argument('--val-annotations',
                        help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--classes',
                        help='Path to a CSV file containing class label mapping.',
                        required=True)
    parser.add_argument('--dest-file',
                        help='Path to destination HDF5 file.',
                        required=True)

    parser.add_argument('--backbone-to-use',
                        help='Backbone that will be used in training.',
                        default='resnet50',
                        type=str)
    parser.add_argument('--image-min-side',
                        help='Rescale the image so the smallest side is min_side.',
                        type=int,
                        default=800)
    parser.add_argument('--image-max-side',
                        help='Rescale the image if the largest side is larger than max_side.',
                        type=int,
                        default=1333)
    parser.add_argument('--no-resize',
                        help='Don\'t rescale the image.',
                        action='store_true')

    args = parser.parse_args()

    return args


def main():
    args = parse()
    annotations_csv = {
        'train': args.train_annotations,
        'val': args.val_annotations,
    }
    classes_csv = args.classes
    dataset_file = args.dest_file

    common_args = {
        'batch_size'       : 1,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : backbone(args.backbone_to_use).preprocess_image,
    }

    transform_generator = None
    visual_effect_generator = None

    for split in ['train', 'val']:
        if not annotations_csv[split]:
            continue

        generator = CSVGenerator(
            annotations_csv[split],
            classes_csv,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        # Computing the data that will be stored
        # H5py does not allow variable length arrays of more than 1 dimension
        # so we save the shapes to be able to reconstruct them.
        # Also preprocessed images are saved so they don't have to be preprocessed avery time they are used in training.
        all_images_group = []
        labels_group = []
        bboxes_group = []
        shapes_group = []

        for i in tqdm(range(generator.size()), desc=f'{split}: '):
            group = [i]
            image_group = generator.load_image_group(group)
            annotations_group = generator.load_annotations_group(group)

            image_group, annotations_group = generator.filter_annotations(image_group, annotations_group, group)
            image_group, annotations_group = generator.preprocess_group(image_group, annotations_group)

            shapes_group += [image_group[0].shape]
            all_images_group += [image_group[0].reshape(-1)]
            labels_group += [annotations_group[0]['labels']]
            bboxes_group += [annotations_group[0]['bboxes'].reshape(-1)]

        save_classes = [k for k in generator.classes]

        # Creating and filling the hdf5 file. We use special dtypes because we have variable lengths in our variables
        dt = h5py.special_dtype(vlen=np.dtype('float64'))
        st = h5py.special_dtype(vlen=str)
        print(f'Saving {split}...')
        with h5py.File(dataset_file, 'a') as hf:
            hf.create_dataset(f'{split}/img', data=all_images_group, compression='gzip', compression_opts=9, dtype=dt)
            hf.create_dataset(f'{split}/shapes', data=shapes_group, compression='gzip', compression_opts=9)
            hf.create_dataset(f'{split}/labels', data=labels_group, compression='gzip', compression_opts=9, dtype=dt)
            hf.create_dataset(f'{split}/bboxes', data=bboxes_group, compression='gzip', compression_opts=9, dtype=dt)
            if split == 'train':
                hf.create_dataset('classes', data=np.string_(save_classes), compression='gzip', compression_opts=9, dtype=st)
        print(f'[OK] {split}')

