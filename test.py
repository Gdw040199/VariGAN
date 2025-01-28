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
    parser.add_argument('--dataroot', type=str, default='dataset/horse2zebra', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='save/dataset/horse2zebra/G_AB_4.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='save/dataset/horse2zebra/G_BA_4.pth',
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

    checkpoint_1 = torch.load(opt.generator_A2B)
    netG_A2B.load_state_dict({k.replace('module.',''):v for k,v in checkpoint_1[opt.generator_A2B].items()})
    checkpoint_2 = torch.load(opt.generator_B2A)
    netG_A2B.load_state_dict({k.replace('module.',''):v for k,v in checkpoint_2[opt.generator_B2A].items()})

    #netG_A2B.load_state_dict(torch.load(opt.generator_A2B), strict=False)
    #netG_B2A.load_state_dict(torch.load(opt.generator_B2A), strict=False)

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
    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')

    for i, batch in enumerate(dataloader):
        ## Input data (real)
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))
        ## Generated data (fake) through the generator
        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)
        ## Save images
        save_image(fake_A, 'output/A/%04d.png' % (i + 1))
        save_image(fake_B, 'output/B/%04d.png' % (i + 1))
        print('processing (%04d)-th image...' % (i))
    print("Testing completed")


if __name__ == '__main__':
    test()  ## Test the model