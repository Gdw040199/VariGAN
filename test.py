import os
import glob
import random
import torch
import itertools
import datetime
import time
import sys
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torchvision.utils import save_image, make_grid
import lpips
from models import Generator
from utils import ReplayBuffer
from datasets import ImageDataset


def test():
    ## Hyperparameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='dataset/facades', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='save/dataset/facades/G_AB_4.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='save/dataset/facades/G_BA_4.pth',
                        help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    #################################
    ##         Test Preparation    ##
    #################################

    ## input_shape: (3, 256, 256)
    input_shape = (opt.channels, opt.size, opt.size)
    ## Create generator and discriminator objects
    netG_A2B = Generator(input_shape, opt.n_residual_blocks)
    netG_B2A = Generator(input_shape, opt.n_residual_blocks)

    ## Use CUDA
    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()

    netG_A2B.load_state_dict(torch.load(opt.generator_A2B), strict=False)
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A), strict=False)

    ## Set to evaluation mode
    netG_A2B.eval()
    netG_B2A.eval()

    ## Create a tensor array
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.channels, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.channels, opt.size, opt.size)

    '''Build the test dataset'''
    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
                            batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    #################################
    ##           Start Test        ##
    #################################

    '''If the file path does not exist, create one (to store test output images)'''
    if not os.path.exists('output/testB'):
        os.makedirs('output/testB', exist_ok=True)
    if not os.path.exists('output/testA'):
        os.makedirs('output/testA', exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        # 数据准备
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # 生成图像（保持梯度计算上下文）
        with torch.no_grad():
            fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)  # [-1,1] -> [0,1]
            fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        # 遍历批次中的每个样本
        batch_size = fake_A.size(0)
        for sample_idx in range(batch_size):
            # 计算全局索引
            global_idx = batch_idx * batch_size + sample_idx

            # 分离并保存单个图像
            single_fakeA = fake_A[sample_idx].unsqueeze(0)  # 保持3D维度
            save_image(single_fakeA,
                       f'output/testB/{global_idx + 1:04d}.png',
                       nrow=1)  # 强制单列显示

            single_fakeB = fake_B[sample_idx].unsqueeze(0)
            save_image(single_fakeB,
                       f'output/testA/{global_idx + 1:04d}.png',
                       nrow=1)

        print(f'Processed batch {batch_idx + 1} ({batch_size} samples)')
    print("Testing completed")


if __name__ == '__main__':
    test()  ## Test the model