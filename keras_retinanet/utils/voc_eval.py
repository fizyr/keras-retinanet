# Pascal VOC Evaluation 2007
# Written by Ashley Williamson

import numpy as np
import pathlib
import cv2

def load_ground_truths( generator, image_id ):
    gt=[]
    for annotation in generator.load_annotations(image_id):
        label = int(annotation[4])
        gt.append({
            'image_id': generator.image_names[image_id],
            'category_id': generator.label_to_name(label), # Use name
            'bbox': (annotation[:4]).tolist()
        })
    return gt

def draw_gt_bboxes(image, anno_list):
    for anno in anno_list:
        label = anno['category_id']
        b = np.array(anno['bbox']).astype(int)

        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        caption = "GT:{}".format(label)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

def filter_top_N_bboxes(bbox_preds, N):
    max_liklihood_sorted = sorted(bbox_preds, key=lambda x: x['score'], reverse=True)
    return max_liklihood_sorted[:min(len(max_liklihood_sorted), N)]

def draw_top_N(image, bbox_preds, generator, N):
    for bbox in filter_top_N_bboxes(bbox_preds, N):
        b = np.array(bbox['bbox']).astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        caption = "P:{} {:.3f}".format(generator.label_to_name(bbox['category_id']), bbox[
            'score'])
        cv2.putText(image, caption, (b[0], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 3)
        cv2.putText(image, caption, (b[0], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

def evaluate_voc(generator, model, threshold=0.05):

    gt = [[] for _ in range(generator.size())]

    for i in range(generator.size()):
        image = generator.load_image(i)
        draw = image.copy()

        image, scale = generator.resize_image(image)
        image = generator.preprocess_image(image)

        _, _, det = model.predict_on_batch(np.expand_dims(image, axis=0))

        gt[i] = load_ground_truths(generator, i)

        det[0, :, :4] /= scale
        det[:, :, 0] = np.maximum(0, det[:, :, 0])
        det[:, :, 1] = np.maximum(0, det[:, :, 1])
        det[:, :, 2] = np.minimum(image.shape[1], det[:, :, 2])
        det[:, :, 3] = np.minimum(image.shape[0], det[:, :, 3])

        results_this_image = []

        for detection in det[0, ...]:
            positive_labels = np.where(detection[4:] > threshold)[0]

            # Skip as we have no positive detections above the threshold for this image
            if len(positive_labels) < 1:
                continue

            # append detections for each positively labeled class
            for indx, label in enumerate(positive_labels):
                image_result = {
                    'image_id': generator.image_names[i],
                    'category_id': label,
                    'score': float(detection[4 + label]),
                    'bbox': (detection[:4]).tolist(),
                }

                results_this_image.append(image_result)

        # Draw Top N BBox predictions!
        draw_top_N(draw, results_this_image, generator, 5)

        # Draw Ground Truth
        draw_gt_bboxes(draw, gt[i])

        dest = pathlib.Path('./images_voc/')
        dest.mkdir(exist_ok=True)
        cv2.imwrite(str(dest.joinpath("{}.jpg".format(i))), draw)

        print('{}/{}'.format(i, len(generator.image_names)), end='\r')