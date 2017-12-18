# Pascal VOC Evaluation 2007
# Written by Ashley Williamson

import numpy as np
import pathlib
import cv2

class VOCEvaluator():
    def __init__(self, generator, model, threshold=0.05, save=False, save_path='images_voc'):
        self.generator = generator
        self.model = model
        self.threshold = threshold
        self.save = save

        self.detections = []

        if self.save:
            self.save_path = pathlib.Path(save_path)
            self.save_path.mkdir(exist_ok=True)

    def load_ground_truths(self, image_id ):
        gt=[]
        for annotation in self.generator.load_annotations(image_id):
            label = int(annotation[4])
            gt.append({
                'image_id': self.generator.image_names[image_id],
                'category_id': self.generator.label_to_name(label), # Use name
                'bbox': (annotation[:4]).tolist()
            })
        return gt

    @staticmethod
    def draw_gt_bboxes(image, anno_list):
        for anno in anno_list:
            label = anno['category_id']
            b = np.array(anno['bbox']).astype(int)

            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
            caption = "GT:{}".format(label)
            cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
            cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    @staticmethod
    def filter_top_N_bboxes(bbox_preds, N):
        max_liklihood_sorted = sorted(bbox_preds, key=lambda x: x['score'], reverse=True)
        return max_liklihood_sorted[:min(len(max_liklihood_sorted), N)]

    @staticmethod
    def preprocess_detection(det, scale, image):
        det[0, :, :4] /= scale
        det[:, :, 0] = np.maximum(0, det[:, :, 0])
        det[:, :, 1] = np.maximum(0, det[:, :, 1])
        det[:, :, 2] = np.minimum(image.shape[1], det[:, :, 2])
        det[:, :, 3] = np.minimum(image.shape[0], det[:, :, 3])

    def draw_top_N(self, image, bbox_preds, N):
        for bbox in self.filter_top_N_bboxes(bbox_preds, N):
            b = np.array(bbox['bbox']).astype(int)
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            caption = "P:{} {:.3f}".format(self.generator.label_to_name(bbox['category_id']), bbox[
                'score'])
            cv2.putText(image, caption, (b[0], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
            cv2.putText(image, caption, (b[0], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    def process_image_detections(self, dets, image_id):
        filtered_detections = []
        for detection in dets[0, :, :]:
            positive_labels = np.where(detection[4:] > self.threshold)[0]

            # Skip as we have no positive detections above the threshold for this image
            if len(positive_labels) < 1:
                continue

            # append detections for each positively labeled class
            for indx, label in enumerate(positive_labels):
                image_result = {
                    'image_id': self.generator.image_names[image_id],
                    'category_id': label,
                    'score': float(detection[4 + label]),
                    'bbox': (detection[:4]).tolist(),
                }

                filtered_detections.append(image_result)

        return filtered_detections

    def save_image(self, image, image_name):
        cv2.imwrite(str(self.save_path.joinpath(image_name)), image)

    def evaluate(self):
        gt = [[] for _ in range(self.generator.size())]

        for i in range(self.generator.size()):

            image = self.generator.load_image(i)
            gt[i] = self.load_ground_truths(i)

            draw = image.copy()

            image, scale = self.generator.resize_image(image)
            image = self.generator.preprocess_image(image)

            _, _, det = self.model.predict_on_batch(np.expand_dims(image, axis=0))

            self.preprocess_detection(det,scale,image)

            results_this_image = self.process_image_detections(det, i)

            # Draw Top N BBox predictions!
            self.draw_top_N(draw, results_this_image, 5)

            # Draw Ground Truth
            self.draw_gt_bboxes(draw, gt[i])

            if self.save:
                self.save_image(draw, "{}.jpg".format(i))

            print('{}/{}'.format(i, self.generator.size()), end='\r')