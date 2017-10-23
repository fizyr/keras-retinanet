# Keras RetinaNet
Keras implementation of RetinaNet object detection as described in [this paper](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Training
An example on how to train `keras-retinanet` can be found [here](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/train_coco.py).

### Usage
For training on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), run:
```
python examples/train_pascal.py <path to VOCdevkit/VOC2007>
```

For training on [MS COCO](http://cocodataset.org/#home), run:
```
python examples/train_coco.py <path to MS COCO>
```

In general, the steps to train on your own datasets are:
1) Create a model by calling `keras_retinanet.models.ResNet50RetinaNet` and compile it. Empirically, the following compile arguments have been found to work well:
```
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.regression_loss,
        'classification': keras_retinanet.losses.focal_loss()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
```
2) Create generators for training and testingdata (an example is show in [`keras_retinanet.preprocessing.PascalVocIterator`](https://github.com/delftrobotics/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)). These generators should generate an image batch (shaped `(batch_id, height, width, channels)`) and a target batch (shaped `(batch_id, num_anchors, 5)`). Currently, a limitation is that `batch_size` must be equal to `1`.
3) Use `model.fit_generator` to start training.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet%20-%20COCO%202017.ipynb). In general, output can be retrieved from the network as follows:
```
_, _, detections = model.predict_on_batch(inputs)
```

Where `detections` are the resulting detections, shaped `(None, None, 4 + num_classes)` (for `(x1, y1, x2, y2, bg, cls1, cls2, ...)`).

Execution time on NVidia Pascal Titan X is roughly 55msec for an image of shape `1000x600x3`.

## Status
* The [examples](https://github.com/delftrobotics/keras-retinanet/tree/master/examples) show how to train `keras-retinanet` on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MS COCO](http://cocodataset.org/). Example output images are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

### Todo's
* Allow `batch_size > 1`.
* Compare result w.r.t. paper results.
* Add unit tests
* Configure CI

### Notes
* This implementation currently uses the `softmax` activation to classify boxes. The paper mentions a `sigmoid` activation instead. Given the origin of parts of this code, the `softmax` activation method was easier to implement. A comparison between `sigmoid` and `softmax` would be interesting, but left as unexplored.
* This repository depends on an unmerged PR of `keras-resnet`. For now, it can be installed by manually installing [this](https://github.com/delftrobotics-forks/keras-resnet/tree/bn-freeze) branch.
* This repository is tested on Keras version 2.0.8, but should also work on 2.0.7.
* This repository is tested using OpenCV 3.3 (3.0+ should be supported).

Any and all contributions to this project are welcome.

### Discussions
Feel free to join the `#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) channel for discussions and questions.
