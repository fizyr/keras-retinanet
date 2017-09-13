# Keras RetinaNet
Keras implementation of RetinaNet object detection as described in [this paper](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Status
* Currently, correct training has been achieved on a simple dataset (malaria dataset, see [this Notebook](https://gist.github.com/hgaiser/c4b29189972c1aa45e99ab9275912910)). Training on more difficult datasets (Pascal VOC) is something left to be explored.
* Training loss and validation loss seem to deviate significantly, further research is required here (possibly causes are overtraining and inconsistency with `BatchNormalization` when training vs testing).
* No postprocessing, such as non-maximum suppression, has been implemented yet.

## Notes
* This implementation currently uses the `softmax` activation to classify boxes. The paper mentions a `sigmoid` activation instead. Given the origin of parts of this code, the `softmax` activation method was easier to implement. A comparison between `sigmoid` and `softmax` would be interesting, but left as unexplored.
* As of writing, this repository depends on an unmerged PR of `keras-resnet`. For now, it can be installed by manually installing [this](https://github.com/delftrobotics-forks/keras-resnet/tree/expose-intermediate) branch.

Any and all contributions to this project are welcome.
