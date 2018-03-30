# import keras
import keras
import sys

from tqdm import tqdm

sys.path.insert(0, 'keras_retinanet')

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, label, score):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.score = score


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def draw_boxes(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 255, 0), 2)
        # cv2.putText(image,
        #             box.label + ' ' + str(box.score),
        #             (box.xmin, box.ymin - 13),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             7e-4 * image.shape[0],
        #             (0, 255, 0), 2)
    return image


# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
model_path = os.path.join('resnet50_coco_best_v2.0.1.h5')

# load retinanet model
model = keras.models.load_model(model_path, custom_objects=custom_objects)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                   5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                   11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog',
                   17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
                   23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
                   29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                   35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                   40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                   47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                   59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                   65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
                   70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                   77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# video_reader = cv2.VideoCapture(
#     '/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/nothing/3.avi')
video_reader = cv2.VideoCapture(
    '/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/кассы 8-9_20171110-192101--20171110-192601.avi')

nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

every_nth = 1
count = 0

pbar = tqdm(total=nb_frames)
while video_reader.isOpened():
    _, image = video_reader.read()
    if image is None:
        break

    count += 1
    pbar.update(1)

    if count % every_nth:
        continue

    draw = image.copy()

    image = preprocess_image(image)
    image, scale = resize_image(image, image.shape[0], image.shape[1])

    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
    detections[0, :, :4] /= scale

    boxes = []
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.3 or label != 0:
            continue

        boxes.append(BoundBox(*detections[0, idx, :4].astype(int), label=labels_to_names[label], score=score))

        draw = draw_boxes(draw, boxes)

    cv2.imshow('Predicted', cv2.resize(draw, (1280, 720)))
    cv2.waitKey(1)


video_reader.release()
pbar.close()
