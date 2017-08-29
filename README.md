# Keras RetinaNet
Keras implementation of RetinaNet object detection as described in [this paper](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Status
* Currently, correct training has been achieved on a very simple dataset (a single black image with a white square on it, see [this Notebook](https://gist.github.com/hgaiser/a2c053ce8dfdd6a829e56c4b3ede5ad3)). 
* In order to get classification to work, the finest pyramid level (`P3` in the paper) was disabled. Presumably this level introduces too many background anchors, causing unstable training. The exact reason and correct fix for this is not yet found. Finding a fix for this has highest priority, since disabling `P3` also disables the ability to detect smaller objects.
* Only classification has been implemented and tested. Regression is mostly in place, but has low priority since it's pointless to try and fix it when classification is not working properly.
* No postprocessing, such as non-maximum suppression, has been implemented yet. Again, this is depending on correctly trained networks, so it has low priority.
* Currently only `python3` is supported, `python2` has low priority. Contributions to fix `python2` support are welcome.
* As of writing, this repository depends on an unmerged PR of `keras-resnet`. For now, it can be installed by manually installing [this](https://github.com/delftrobotics-forks/keras-resnet/tree/expose-intermediate) branch.

Any and all contributions to this project are welcome.
