import os
import time
import json
from glob import glob
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from labelme import utils  # pip install labelme
from EdgeFenceSeg import EdgeFenceSeg

class LabelmeParser:
    def __init__(self, image_shape):
        self.height, self.width = image_shape[:2]

    def json_to_mask(self, json_path):
        """将LabelMe标注文件转换为二值掩膜"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # build mask
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # read all polygons
        for shape in data['shapes']:
            points = np.array(shape['points'], dtype=np.int32)
            #  change  shape to masks
            shape_mask = utils.shape_to_mask(
                (self.height, self.width),
                points,
                shape_type=shape['shape_type']
            )
            mask = np.bitwise_or(mask, shape_mask.astype(np.uint8) * 255)

        return mask


class MetricsCalculator:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, y_true, y_pred):
        y_true = y_true.flatten().astype(np.int32)
        y_pred = y_pred.flatten().astype(np.int32)
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        self.total_cm += cm

    def pixel_accuracy(self):
        return np.diag(self.total_cm).sum() / self.total_cm.sum()

    def class_accuracy(self):
        return np.nanmean(np.diag(self.total_cm) / self.total_cm.sum(axis=1))

    def class_iou(self):
        intersection = np.diag(self.total_cm)
        union = self.total_cm.sum(axis=1) + self.total_cm.sum(axis=0) - intersection
        return np.nanmean(intersection / union)


# main
if __name__ == "__main__":
    # parameters (roots)
    imgs_path = "dataset/imgs"
    label_path = "dataset/jsons"  # the path of labelme json files
    output_path = "output_cv"

    # initialize
    metrics = MetricsCalculator()
    edg = EdgeFenceSeg()

    # read all the images and labels
    image_files = sorted(glob(os.path.join(imgs_path, '*.png')))
    for img_file in image_files:

        img = cv2.imread(img_file)
        base_name = os.path.basename(img_file).split('.')[0]

        json_file = os.path.join(label_path, base_name + '.json')
        if not os.path.exists(json_file):
            print(f"Warning: Missing label for {base_name}")
            continue

        # build mask from labelme json
        parser = LabelmeParser(img.shape)
        label_mask = parser.json_to_mask(json_file)

        # predict
        pred_mask, prediction, flag = edg.predict(img)
        if pred_mask is None:
            continue

        pred_mask, prediction, flag = edg.predict(img)
        if pred_mask is None:
            continue

        # change the shape of the prediction as the edgefenceseg.py returned PIL image
        pred_mask = np.array(pred_mask)
        if pred_mask.ndim == 3:
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2GRAY)

        # resize the prediction to the same size as the label
        pred_mask = cv2.resize(pred_mask, (label_mask.shape[1], label_mask.shape[0]))

        # binarize the mask
        _, pred_bin = cv2.threshold(pred_mask, 127, 1, cv2.THRESH_BINARY)
        _, label_bin = cv2.threshold(label_mask, 127, 1, cv2.THRESH_BINARY)

        # update metrics
        metrics.update(label_bin, pred_bin)

        cv2.imwrite(os.path.join(output_path, base_name + '_mask.png'), pred_mask * 255)

    # output metrics
    print(f"\nFinal Evaluation:")
    print(f"Pixel Accuracy: {metrics.pixel_accuracy():.4f}")
    print(f"Class Accuracy: {metrics.class_accuracy():.4f}")
    print(f"Mean IoU: {metrics.class_iou():.4f}")
