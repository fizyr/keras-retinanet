# Keras RetinaNet
Keras implementation of RetinaNet object detection as described in [this paper](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Training
An example on how to train `keras-retinanet` can be found [here](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/train.py).

### Usage
For training on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), run:
```
python examples/train.py <path to VOCdevkit/VOC2007>
```

In general, the steps to train on your own datasets are:
1) Create a model by calling `keras_retinanet.models.ResNet50RetinaNet` and compile it. Empirically, the following compile arguments have been found to work well:
```
model.compile(loss=None, optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))
```
2) Create generators for training and testing data (an example is show in [`keras_retinanet.preprocessing.PascalVocIterator`](https://github.com/delftrobotics/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)). These generators should generate an image batch (shaped `(batch_id, height, width, channels)`) and a boxes batch (shaped `(batch_id, num_boxes, 5)`, where the last dimension is for `(x1, y1, x2, y2, label)`). Currently, a limitation is that `batch_size` must be equal to `1` and the image shape must be defined beforehand (ie. it does _not_ accept input images of shape `(None, None, 3)`).
3) Use `model.fit_generator` to start training.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet%20-%20Pascal%20VOC.ipynb). In general, output can be retrieved from the network as follows:
```
boxes, classification, reg_loss, cls_loss = model.predict_on_batch(inputs)
```

Where `boxes` are the resulting bounding boxes, shaped `(None, 4)` (for `(x1, y1, x2, y2)`). `classification` is the corresponding class scores for each box (shaped `(None, num_classes)`). `reg_loss` is the regression loss value and `cls_loss` is the classification loss value.

Execution time on NVidia Pascal Titan X is roughly 35msec for an image of shape `512x512x3`.

## Status
* The [examples](https://github.com/delftrobotics/keras-retinanet/tree/master/examples) show how to train `keras-retinanet` on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) data. An example output image is shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/pascal_voc.png" alt="Example result of RetinaNet on Pascal VOC"/>
</p>

### Todo's
* Allow `batch_size > 1`.
* Compare result w.r.t. paper results.
* Disable parts of the network when in test mode.
* Fix saving / loading of model (currently only saving / loading of weights works).

### Notes
* This implementation currently uses the `softmax` activation to classify boxes. The paper mentions a `sigmoid` activation instead. Given the origin of parts of this code, the `softmax` activation method was easier to implement. A comparison between `sigmoid` and `softmax` would be interesting, but left as unexplored.
* As of writing, this repository depends on an unmerged PR of `keras-resnet`. For now, it can be installed by manually installing [this](https://github.com/delftrobotics-forks/keras-resnet/tree/expose-intermediate) branch.

Any and all contributions to this project are welcome.
