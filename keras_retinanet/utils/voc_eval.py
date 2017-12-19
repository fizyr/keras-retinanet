# Pascal VOC Evaluation 2007
# Written by Ashley Williamson

import numpy as np
import pathlib
import cv2

from keras_retinanet.utils.anchors import compute_overlap


class VOCEvaluator():
    def __init__(self, generator, model, threshold=0.05, iou_threshold=0.5, max_detections=100, save=False,
                 save_path='images_voc'):
        self.generator = generator
        self.model = model
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.save = save

        self.detections = [[None for _ in range(self.generator.num_classes())] for _ in range(self.generator.size())]
        self.ground_truth = [[None for _ in range(self.generator.num_classes())] for _ in range(self.generator.size())]

        self.average_precisions = np.zeros((0,))

        if self.save:
            self.save_path = pathlib.Path(save_path)
            self.save_path.mkdir(exist_ok=True)

    def load_ground_truths(self):
        for i in range(self.generator.size()):
            annotations = self.generator.load_annotations(i)

            for l in range(self.generator.num_classes()):
                self.ground_truth[i][l] = annotations[annotations[:, 4] == l, :4].copy()

    @staticmethod
    def _compute_ap(recall, precision):
        # code originally from https://github.com/rbgirshick/py-faster-rcnn

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @staticmethod
    def draw_bbox(image, bbox, label, gt=False):
        bbox = np.array(bbox).astype(int)
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0) if gt else (0,0,255), 1)
        caption = "{}: {}".format("GT" if gt else "P", label)
        cv2.putText(image, caption, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
        cv2.putText(image, caption, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    @staticmethod
    def preprocess_detection(det, scale, image):
        det[0, :, :4] /= scale
        det[:, :, 0] = np.maximum(0, det[:, :, 0])
        det[:, :, 1] = np.maximum(0, det[:, :, 1])
        det[:, :, 2] = np.minimum(image.shape[1], det[:, :, 2])
        det[:, :, 3] = np.minimum(image.shape[0], det[:, :, 3])

    def save_image(self, image, image_id):
        if self.save:
            cv2.imwrite(str(self.save_path.joinpath("{}.jpg".format(image_id))), image)

    def populate_detections(self):
        for i in range(self.generator.size()):
            image = self.generator.load_image(i)
            if self.save:
                draw = image.copy()

            image = self.generator.preprocess_image(image)
            image, scale = self.generator.resize_image(image)

            _, _, det = self.model.predict_on_batch(np.expand_dims(image, axis=0))

            # Rescale and clip detection bboxes to new image.
            self.preprocess_detection(det, scale, image)

            # Obtain indices for scores where they are over the score threshold.
            chosen_scores_indices = np.where(det[0,:,4:] > self.threshold)

            # Actually grab those indices.
            actual_scores = det[0,:,4:][chosen_scores_indices]

            # Sort, clipping to max detection number.
            sorted_scores_order = np.argsort(-actual_scores)[:self.max_detections]

            # Pick the bbox from the sorted, cropped list, ignoring the score component.
            image_boxes         = det[0, chosen_scores_indices[0][sorted_scores_order], :4]

            # Score component for all labels.
            image_scores        = np.expand_dims(det[0, chosen_scores_indices[0][sorted_scores_order],
                                                  4 + chosen_scores_indices[1][
                sorted_scores_order]], axis=1)

            # [image_boxes] + [scores]
            image_det           = np.append(image_boxes, image_scores, axis=1) # Tack score at end of each bbox.

            image_pred_label    = chosen_scores_indices[1][sorted_scores_order]

            for l in range(self.generator.num_classes()):
                # Bin the detections into their respective classes for this image.
                self.detections[i][l] = image_det[image_pred_label == l, :]

            if self.save:
                for ind, d in enumerate(image_det):
                    self.draw_bbox(draw, d[:4], self.generator.label_to_name(image_pred_label[ind]), gt=False)
                for label, bboxes in enumerate(self.ground_truth[i]):
                    for b in bboxes:
                        self.draw_bbox(draw, b[:4], self.generator.label_to_name(label), gt=True)
                self.save_image(draw,i)

            print('{}/{}'.format(i, self.generator.size()), end='\r')


    def evaluate(self):

        # Initialise Ground Truth data
        self.load_ground_truths()

        #Initialise Detections
        self.populate_detections()

        for l in range(self.generator.num_classes()):
            tp      = np.zeros((0,))
            fp      = np.zeros((0,))
            scores  = np.zeros((0,))
            npos = 0

            for i in range(self.generator.size()):
                dets    = self.detections[i][l]
                gt      = self.ground_truth[i][l]
                npos += gt.shape[0]
                detected_record = []

                for detection in dets:

                    # Append the score first as we might skip!
                    scores = np.append(scores, detection[4])

                    # Ensure shape.
                    if gt.shape[0] == 0:
                        fp = np.append(fp, 1)
                        tp = np.append(tp, 0)
                        continue

                    #Calculate overlap
                    overlaps = compute_overlap(np.expand_dims(detection, axis=0), gt)
                    best_overlap = np.argmax(overlaps, axis=1)

                    if best_overlap not in detected_record and overlaps[0, best_overlap] > self.iou_threshold:
                        # Only capture once
                        detected_record.append(best_overlap)

                        # Positive count.
                        fp = np.append(fp, 0)
                        tp = np.append(tp, 1)
                    else:
                        # Negative count.
                        fp = np.append(fp, 1)
                        tp = np.append(tp, 0)


            indices = np.argsort(-scores)
            tp = tp[indices]
            fp = fp[indices]

            tp_score = np.cumsum(tp)
            fp_score = np.cumsum(fp)

            recall = tp_score / npos
            precision = tp_score /  np.maximum(tp_score + fp_score, np.finfo(np.float64).eps)

            ap = self._compute_ap(recall,precision)
            self.average_precisions = np.append(self.average_precisions, ap)

            print(self.generator.label_to_name(l), ap)
        print("mAP: {}".format(self.average_precisions.mean()))