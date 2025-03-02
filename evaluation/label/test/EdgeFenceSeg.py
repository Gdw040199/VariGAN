# -*- coding: utf-8 -*-
"""
Created on 2020.01.12
@author: LWS
Inference of Edge Fence Segmentation.
"""

import numpy as np
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.deeplabv3_plus import DeepLab
from utils.helpers import colorize_mask
from utils import palette


class EdgeFenceSeg(object):

    def __init__(self,
                 model_path="ckpt/best_model.pth",
                 area_thres=0.1,
                 cuda_id=0,
                 get_mask=True):

        torch.set_num_threads(8)

        self.area_thres = area_thres
        self.num_classes = 2  # background + fence
        self.get_mask = get_mask

        # get palette of VOC
        self.my_palette = palette.get_voc_palette(self.num_classes)

        # data setting
        self._MEAN = [0.48311856, 0.49071315, 0.45774156]
        self._STD = [0.21628413, 0.22036915, 0.22477823]
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self._MEAN, self._STD)

        # get Model
        self.model = DeepLab(num_classes=self.num_classes, backbone='resnet101', pretrained=False)
        availble_gpus = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:{}'.format(cuda_id) if len(availble_gpus) > 0 else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        # If during training, we used data parallel
        if ('module' in list(checkpoint.keys())[0] and
                not isinstance(self.model, torch.nn.DataParallel)):
            # for gpu inference, use data parallel
            if "cuda" in self.device.type:
                self.model = torch.nn.DataParallel(self.model)
            else:
                # for cpu inference, remove module
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]
                    new_state_dict[name] = v
                checkpoint = new_state_dict
        # load
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, img):
        """
        :param img: image for predict, np.ndarray.
        :return: mask_img, prediction, flag;
        if all None, means image type error; if mask_img is None, means don't extract mask.
        """
        if str(type(img)) == "<class 'NoneType'>":
            return None, None, None
        flag = False
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        with torch.no_grad():
            input = self.normalize(self.to_tensor(img)).unsqueeze(0)
            prediction = self.model(input.to(self.device))
            prediction = prediction.squeeze(0).cpu().numpy()
            prediction = F.softmax(torch.from_numpy(prediction),
                                   dim=0).argmax(0).cpu().numpy()
            area_ratio = sum(prediction[prediction == 1]) / (img.size[0] * img.size[1])
            if area_ratio >= self.area_thres:
                flag = True
        if self.get_mask:
            mask_img = self.colored_mask_img(img, prediction)
            return mask_img, prediction, flag
        else:
            return None, prediction, flag

    def colored_mask_img(self, image, mask):
        colorized_mask = colorize_mask(mask, self.my_palette)
        # PIL type
        mask_img = Image.blend(image.convert('RGBA'), colorized_mask.convert('RGBA'), 0.7)
        return mask_img
