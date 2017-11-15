# Keras RetinaNet
Keras implementation of RetinaNet object detection as described in [this paper](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Installation

1) Clone this repository.
2) In the repository, execute `pip install .`. Note that due to inconsistencies with how `tensorflow` should be installed, this package does not define a dependency on `tensorflow` as it will try to install that through `pip` (which at least on Arch linux results in an incorrect installation). Please make sure `tensorflow` is installed as per your systems requirements. Also, make sure Keras 2.0.9 or above is installed as this package uses some features of 2.0.9.
3) As of writing, this repository requires the `master` version of `keras-resnet` for freezing `BatchNormalization` layers (ie. clone [this](https://github.com/broadinstitute/keras-resnet) repository and run `pip install .` in that repository).
4) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset. Clone the [`cocoapi` repository](https://github.com/cocodataset/cocoapi) and inside the `PythonAPI` folder, execute `pip install .`.

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

For training on a custom dataset, a CSV file can be used as a way to pass the data. To train using your CSV, run:
```
python examples/train_csv.py <path to csv file containing annotations> <path to csv file containing classes>
```

The expected format of each line of the annotations CSV is:
```
filepath,x1,y1,x2,y2,class_name
```

For example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
```
Note that indexing for pixel values starts at 0. The expected format of each line of the classes CSV is:
```
class_name,id
```

For example:
```
cow,0
cat,1
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
2) Create generators for training and testing data (an example is show in [`keras_retinanet.preprocessing.PascalVocIterator`](https://github.com/delftrobotics/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)). These generators should generate an image batch (shaped `(batch_id, height, width, channels)`) and a target batch (shaped `(batch_id, num_anchors, 4 + num_classes)`). Currently, a limitation is that `batch_size` must be equal to `1`.
3) Use `model.fit_generator` to start training.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet%20-%20COCO%202017.ipynb). In general, output can be retrieved from the network as follows:
```
_, _, detections = model.predict_on_batch(inputs)
```

Where `detections` are the resulting detections, shaped `(None, None, 4 + num_classes)` (for `(x1, y1, x2, y2, cls1, cls2, ...)`).

Execution time on NVIDIA Pascal Titan X is roughly 55msec for an image of shape `1000x600x3`.

## Results

### MS COCO
The MS COCO model can be downloaded [here](https://delftrobotics-my.sharepoint.com/personal/h_gaiser_fizyr_com/_layouts/15/guestaccess.aspx?docid=08db522fc87ba48189ca4629377c2646e&authkey=AS3wwMzO1bb9dHBvrMdNJ5s&e=cde447bbbd2d42ad88b5a5d9d623bbfa). Results using the `cocoapi` are shown below (note: according to the paper, this configuration should achieve a mAP of 0.34).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.485
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.587
```

## Status
* The [examples](https://github.com/delftrobotics/keras-retinanet/tree/master/examples) show how to train `keras-retinanet` on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [MS COCO](http://cocodataset.org/). Example output images are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

### Todo's
* Allow `batch_size > 1`.
* Configure CI

### Notes
* This repository requires Keras 2.0.9 or above.
* This repository is tested using OpenCV 3.3 (3.0+ should be supported).

Contributions to this project are welcome.

### Discussions
Feel free to join the `#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) channel for discussions and questions.
