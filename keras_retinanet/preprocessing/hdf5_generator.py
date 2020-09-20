from collections import OrderedDict

import h5py

from .generator import Generator


class HDF5Generator(Generator):
    def __init__(
            self,
            hdf5_file,
            partition,
            **kwargs
    ):
        with h5py.File(hdf5_file, 'r') as hf:
            self.images = list(hf[partition]['img'])
            shapes = list(hf[partition]['shapes'])
            self.labels = list(hf[partition]['labels'])
            self.bboxes = list(hf[partition]['bboxes'])
            self.classes = list(hf['classes'])

        # hdf5 only allows storage of unidimensional arrays if they have different lengths
        self.images = [img.reshape(shapes[i]) for i, img in enumerate(self.images)]
        self.bboxes = [box.reshape(-1, 4) for box in self.bboxes]
        self.classes = OrderedDict({key: i for i, key in enumerate(self.classes)})
        super(HDF5Generator, self).__init__(**kwargs)

    def size(self):
        return len(self.images)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        return float(self.images[image_index].shape[1]) / float(self.images[image_index].shape[0])

    def get_image_group(self, group):
        return [self.images[i] for i in group]

    def get_annotations_group(self, group):
        return [{'labels': self.labels[i],
                 'bboxes': self.bboxes[i]} for i in group]

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group = self.get_image_group(group)
        annotations_group = self.get_annotations_group(group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets