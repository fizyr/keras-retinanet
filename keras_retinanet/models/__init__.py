import keras.models
from .retinanet import retinanet_bbox


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

    model = keras.models.load_model(filepath, custom_objects=custom_objects(backbone))
    if convert:
        model = retinanet_bbox(model=model, nms=nms)

    return model


def custom_objects(backbone):
    if 'resnet' in backbone:
        from .resnet import custom_objects as co
    elif 'mobilenet' in backbone:
        from .mobilenet import custom_objects as co
    elif 'vgg' in backbone:
        from .vgg import custom_objects as co
    elif 'densenet' in backbone:
        from .densenet import custom_objects as co
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(backbone))

    return co


def retinanet_backbone(backbone):
    if 'resnet' in backbone:
        from .resnet import resnet_retinanet as rn
    elif 'mobilenet' in backbone:
        from .mobilenet import mobilenet_retinanet as rn
    elif 'vgg' in backbone:
        from .vgg import vgg_retinanet as rn
    elif 'densenet' in backbone:
        from .densenet import densenet_retinanet as rn
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(backbone))

    return rn


def download_imagenet(backbone):
    if 'resnet' in backbone:
        from .resnet import download_imagenet as di
    elif 'mobilenet' in backbone:
        from .mobilenet import download_imagenet as di
    elif 'vgg' in backbone:
        from .vgg import download_imagenet as di
    elif 'densenet' in backbone:
        from .densenet import download_imagenet as di
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(backbone))

    return di(backbone)


def validate_backbone(backbone):
    if 'resnet' in backbone:
        from .resnet import validate_backbone as vb
    elif 'mobilenet' in backbone:
        from .mobilenet import validate_backbone as vb
    elif 'vgg' in backbone:
        from .vgg import validate_backbone as vb
    elif 'densenet' in backbone:
        from .densenet import validate_backbone as vb
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(backbone))

    return vb(backbone)
