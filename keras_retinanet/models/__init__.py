class Backbone(object):
    """ This class stores meta data on backbones.
    """
    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        from .. import layers
        from .. import losses
        from .. import initializers
        self.custom_objects = {
            'UpsampleLike'     : layers.UpsampleLike,
            'PriorProbability' : initializers.PriorProbability,
            'RegressBoxes'     : layers.RegressBoxes,
            'FilterDetections' : layers.FilterDetections,
            'Anchors'          : layers.Anchors,
            'ClipBoxes'        : layers.ClipBoxes,
            '_smooth_l1'       : losses.smooth_l1(),
            '_focal'           : losses.focal(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')


def backbone_meta(backbone):
    """ Returns a backbone meta object for the given backbone.
    """
    if 'resnet' in backbone:
        from .resnet import ResNetBackbone as b
    elif 'mobilenet' in backbone:
        from .mobilenet import MobileNetBackbone as b
    elif 'vgg' in backbone:
        from .vgg import VGGBackbone as b
    elif 'densenet' in backbone:
        from .densenet import DenseNetBackbone as b
    else:
        raise NotImplementedError('Backbone meta class fo  \'{}\' not implemented.'.format(backbone))

    return b(backbone)


def load_model(filepath, backbone='resnet50', convert=False, nms=True):
    """ Loads a retinanet model using the correct custom objects.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone: Backbone with which the model was trained.
        convert: Boolean, whether to convert the model to an inference model.
        nms: Boolean, whether to add NMS filtering to the converted model. Only valid if convert=True.

    # Returns
        A keras.models.Model object.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models

    model = keras.models.load_model(filepath, custom_objects=backbone_meta(backbone).custom_objects)
    if convert:
        from .retinanet import retinanet_bbox
        model = retinanet_bbox(model=model, nms=nms)

    return model
