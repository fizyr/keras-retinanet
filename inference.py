import keras
import sys

from tqdm import tqdm

sys.path.insert(0, 'keras_retinanet')

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image, resize_image
import cv2
import os
import numpy as np
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
        cv2.putText(image,
                    box.label + ' ' + str(box.score),
                    (box.xmin, box.ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    7e-4 * image.shape[0],
                    (0, 255, 0), 2)
    return image

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = ""

keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
model_path = os.path.join('snapshots/resnet50_trassir_02.h5')
# model_path = os.path.join('snapshots/resnet50_trassir_31.h5')
# model_path = os.path.join('resnet50_coco_best_v2.0.1.h5')

# load retinanet model
model = keras.models.load_model(model_path, custom_objects=custom_objects)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'background', 1: 'person'}

# video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/abandonment/rzd2/nothing/3.avi')
# video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/кассы 8-9_20171110-192101--20171110-192601.avi')
video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/очереди/Lanser 3MP-16 10_20171110-193448--20171110-194108.avi')
# video_reader = cv2.VideoCapture('/media/oem/022cfb2b-3c52-4dfe-a5fb-c5fe826db5e3/samples/queues/КАССА 5_20140825-190247--20140825-220330.tva')

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
    image, scale = resize_image(image)

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
