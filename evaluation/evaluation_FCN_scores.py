import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os


class SegmentationEvaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def _fast_hist(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)
        hist = np.bincount(
            self.num_classes * target[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, pred, target):
        self.confusion_matrix += self._fast_hist(pred.flatten(), target.flatten())

    def get_pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def get_class_accuracy(self):
        acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        return acc.mean()

    def get_iou(self):
        iou = np.diag(self.confusion_matrix) / (
                self.confusion_matrix.sum(axis=1) +
                self.confusion_matrix.sum(axis=0) -
                np.diag(self.confusion_matrix) + 1e-10
        )
        return iou.mean()


def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    batch_size = 4

    # load the FCN model
    model = models.segmentation.fcn_resnet50(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.load_state_dict(torch.load("custom_fcn.pth"))
    model = model.to(device).eval()

    # the initial preprocessing step for the input images
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # initialize the evaluator
    evaluator = SegmentationEvaluator(num_classes)

    # path to the generated images and ground truth labels
    gen_images_dir = "path/to/generated_images"
    gt_labels_dir = "path/to/ground_truth_labels"

    # iterate over the generated images
    for img_name in os.listdir(gen_images_dir):
        # load the generated image and the corresponding ground truth label
        gen_img_path = os.path.join(gen_images_dir, img_name)
        label_path = os.path.join(gt_labels_dir, img_name.replace(".jpg", ".png"))

        # load the generated image and convert it to a tensor
        img = Image.open(gen_img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # load the ground truth label
        label = np.array(Image.open(label_path).resize((256, 256)))

        # forward pass
        with torch.no_grad():
            output = model(img_tensor)['out']
        pred = output.argmax(1).squeeze(0).cpu().numpy()

        # update the evaluator
        evaluator.update(pred, label)

    # calculate the evaluation metrics
    pixel_acc = evaluator.get_pixel_accuracy()
    class_acc = evaluator.get_class_accuracy()
    mean_iou = evaluator.get_iou()

    print(f"Per-pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean Class Accuracy: {class_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    main()
