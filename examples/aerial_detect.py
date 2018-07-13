# import keras and tensorflow backend
import keras
import tensorflow as tf

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import csv

# set tf backend to allow memory to grow instead of claiming everything
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# choose which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set modified tf session as  backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# load model from path
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
model = models.load_model(model_path, backbone_name='resnet50')
print(model.summary())

# COCO label to names mapping
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# generate paths to the desired images
image_dir_path = os.path.join('..', '..', '..', 'image_data', 'Collection5')
image_filenames = [filename for filename in os.listdir(image_dir_path) if filename.endswith('.png')]
image_filenames.sort(key=lambda f: f.split('__')[1])

# load CTIL annotations, row is formatted as follows
# image name, object1 name, object1 xmin, object1 ymin, object1 xmax, object1 ymax, object2 name, object2 xmin ... etc
annotations = dict()
with open(os.path.join(image_dir_path, 'ctil_annotations.csv')) as csvfile:
    csvfile.readline()
    reader = csv.reader(csvfile)
    for row in reader:
        # fill dictionary entry for image name with object bounding box info
        objects = np.reshape(row[1:], (-1,5)).tolist()
        for o in range(len(objects)):
            for i in range(4):
                objects[o][i+1] = int(objects[o][i+1])
        annotations[row[0]] = objects


iou_threshold = 0.5 # HYPERPARAMETER TO BE CHANGED

prediction_thresholds = np.arange(.1,.95,.05)
precisions, recalls = np.zeros(len(prediction_thresholds)), np.zeros(len(prediction_thresholds))

# get intersection over union of two boxes
def get_iou(box1, box2):
    # get intersecting rectangle coordinates
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou

# record results in a csv file
with open('test_precision_recall.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Prediction Threshold', 'True Positives', 'Total Predictions', 'Total Objects', 'Precision', 'Recall'])

    # calculate precision and recall values for each prediction_threshold
    for prediction_threshold in prediction_thresholds:

        # count total predictions = true positives + false positives
        total_predictions = 0
        true_positives = 0
        total_objects = 0

        # iterate through all images
        # image_filenames = image_filenames[:4]
        for image_filename in image_filenames:
            print(image_filename)

            # load ground truth annotations
            objects = annotations[image_filename]
            total_objects += len(objects)

            # load and preprocess image
            image = read_image_bgr(os.path.join(image_dir_path, image_filename))
            image = preprocess_image(image)
            image, scale = resize_image(image)

            # forward image through RetinaNet and retrieve bounding boxes, confidence scores, and labels
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            boxes /= scale

            # 1 if corresponding object has been detected
            detected = np.zeros(len(objects))

            # iterate through all network detections
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # sorted in decreasing order of confidence score, can skip rest of detections when score is below prediction_threshold
                if score < prediction_threshold:
                    break
                if labels_to_names[label] != 'car' and labels_to_names[label] != 'traffic light':
                    continue
                total_predictions += 1

                # calculate IOUs between detection and each ground truth box and filter based on IOU threshold
                b = box.astype(int)
                ious = [get_iou(b, object[1:]) for object in objects]
                for i in range(len(ious)):
                    iou = ious[i]
                    if iou >= iou_threshold and labels_to_names[label] == objects[i][0]:
                        detected[i] = 1
            true_positives += np.sum(detected)

        # calculate precision and recall
        precision = true_positives/total_predictions
        recall = true_positives/total_objects

        # write to csv file
        writer.writerow([prediction_threshold, true_positives, total_predictions, total_objects, precision, recall])

        # print to terminal
        print('prediction_threshold =', prediction_threshold)
        print('true positives', true_positives)
        print('total predictions', total_predictions)
        print('total objects', total_objects)
        print('precision', precision)
        print('recall', recall)
        print()
