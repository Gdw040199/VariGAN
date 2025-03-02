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
from models import Generator, Discriminator
from utils import ReplayBuffer, LambdaLR
from datasets import ImageDataset

###########################################################################
############################  cursor.py  ###################################

# Hyperparameter configuration
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="dataset/vangogh2photo",
                    help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
# opt = parser.parse_args(args=[])                 ## Use this line when running in colab
print(opt)

# Create directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("save/%s" % opt.dataset_name, exist_ok=True)

# input_shape:(3, 256, 256)
input_shape = (opt.channels, opt.img_height, opt.img_width)

# Create generator and discriminator objects
G_AB = Generator(input_shape, opt.n_residual_blocks)
G_BA = Generator(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

# Loss functions
# MES: Binary cross-entropy
# L1loss: Preserves edges better compared to L2 Loss
criterion_GAN = torch.nn.MSELoss()
loss_fn_lpips = lpips.LPIPS(net='alex')
criterion_identity = torch.nn.L1Loss()

# If GPU is available, run in CUDA mode
if torch.cuda.is_available():
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    loss_fn_lpips.cuda()
    criterion_identity.cuda()

# If epoch == 0, initialize model parameters; if epoch == n, load the pre-trained model trained up to the nth epoch
if opt.epoch != 0:
    # Load the pre-trained model trained up to the nth epoch
    G_AB.load_state_dict(torch.load("save/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load("save/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("save/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("save/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))


# Define optimization function, learning rate is 0.0003
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update process
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

# Buffer for previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12)),  # Resize image by 1.12 times
    transforms.RandomCrop((opt.img_height, opt.img_width)),  # Randomly crop to original size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize
]

# Training data loader
dataloader = DataLoader(  # Change to your own directory
    ImageDataset("dataset/vangogh2photo", transforms_=transforms_, unaligned=True),
    # "./datasets/facades" , unaligned: Set unaligned data
    batch_size=opt.batch_size,  # batch_size = 1
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("dataset/vangogh2photo", transforms_=transforms_, unaligned=True, mode="test"),  # "./datasets/facades"
    batch_size=5,
    shuffle=True,
    num_workers=1,
)


# Save generated samples from the test set every 100 iterations
def sample_images(batches_done):  # (100/200/300/400...)
    """Save generated samples from the test set"""
    imgs = next(iter(val_dataloader))  # Take one image
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"]).cuda()  # Take a real A
    fake_B = G_AB(real_A)  # Generate fake B from real A
    real_B = Variable(imgs["B"]).cuda()  # Take a real B
    fake_A = G_BA(real_B)  # Generate fake A from real B
    # Arrange images along x-axis
    # make_grid(): Used to arrange several images in a grid
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arrange images along y-axis
    # Concatenate all images and save as one large image
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


def train():
    # ----------
    #  Training
    # ----------
    prev_time = time.time()  # Start time
    for epoch in range(opt.epoch, opt.n_epochs):  # for epoch in (0, 50)
        for i, batch in enumerate(
                dataloader):  # batch is a dict, batch['A']:(1, 3, 256, 256), batch['B']:(1, 3, 256, 256)
            #       print('here is %d' % i)
            # Read real images from the dataset
            # Convert tensor to Variable for computation graph, tensor must be converted to variable for backpropagation
            real_A = Variable(batch["A"]).cuda()  # Real image A
            real_B = Variable(batch["B"]).cuda()  # Real image B

            # Labels for real and fake images
            valid = Variable(torch.ones((real_A.size(0), *D_A.output_shape)),
                             requires_grad=False).cuda()  # Define real image label as 1 ones((1, 1, 16, 16))
            fake = Variable(torch.zeros((real_A.size(0), *D_A.output_shape)),
                            requires_grad=False).cuda()  # Define fake image label as 0 zeros((1, 1, 16, 16))

            ## -----------------
            ##  Train Generator
            ## Principle: The goal is to make the generated fake images be judged as real by the discriminator.
            ## During this process, the discriminator is fixed, and the fake images are passed to the discriminator.
            ## The result is compared with the real label.
            ## The parameters updated through backpropagation are those in the generator.
            ## This way, the generator is trained to produce images that the discriminator thinks are real, achieving the adversarial goal.
            ## -----------------
            G_AB.train()
            G_BA.train()

            ## Identity loss                                              ## A-style image placed in the B -> A generator should also produce A-style images
            loss_id_A = criterion_identity(G_BA(real_A),
                                           real_A)  ## loss_id_A is the loss when image A1 is passed through the B2A generator. The generated image A2 should also be in A-style, and the difference between A1 and A2 should be small.
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2  ## Identity loss

            ## GAN loss
            fake_B = G_AB(real_A)  ## Generate fake image B from real image A
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)  ## Use discriminator B to judge fake image B. The goal of training the generator is to make the discriminator think the fake is real.
            fake_A = G_BA(real_B)  ## Generate fake image A from real image B
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)  ## Use discriminator A to judge fake image A. The goal of training the generator is to make the discriminator think the fake is real.

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2  ## GAN loss

            # Cycle Consistency Loss with LPIPS
            recov_A = G_BA(fake_B)
            loss_cycle_A = loss_fn_lpips(recov_A, real_A).mean()  # Use LPIPS to calculate perceptual loss
            recov_B = G_AB(fake_A)
            loss_cycle_B = loss_fn_lpips(recov_B, real_B).mean()  # Use LPIPS to calculate perceptual loss

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2  # Take the average

            # Total loss                                                  ## Sum of all losses above
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity
            optimizer_G.zero_grad()  ## Zero the gradients before backpropagation
            loss_G.backward()  ## Backpropagate the error
            optimizer_G.step()  ## Update parameters

            ## -----------------------
            ## Train Discriminator A
            ## Divided into two parts: 1. Judge real images as real; 2. Judge fake images as fake
            ## -----------------------
            ## Judge real images as real
            loss_real = criterion_GAN(D_A(real_A), valid)
            ## Judge fake images as fake (randomly take one from the previous buffer)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            optimizer_D_A.zero_grad()  ## Zero the gradients before backpropagation
            loss_D_A.backward()  ## Backpropagate the error
            optimizer_D_A.step()  ## Update parameters

            ## -----------------------
            ## Train Discriminator B
            ## Divided into two parts: 1. Judge real images as real; 2. Judge fake images as fake
            ## -----------------------
            # Judge real images as real
            loss_real = criterion_GAN(D_B(real_B), valid)
            ## Judge fake images as fake (randomly take one from the previous buffer)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2
            optimizer_D_B.zero_grad()  ## Zero the gradients before backpropagation
            loss_D_B.backward()  ## Backpropagate the error
            optimizer_D_B.step()  ## Update parameters
            loss_D = (loss_D_A + loss_D_B) / 2

            ## ----------------------
            ##  Log Progress
            ## ----------------------

            ## Determine the approximate remaining time. Assume current epoch = 5, i = 100
            batches_done = epoch * len(dataloader) + i  ## Time already trained: 5 * 400 + 100 iterations
            batches_left = opt.n_epochs * len(dataloader) - batches_done  ## Remaining iterations: 50 * 400 - 2100
            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time))  ## Time left = remaining iterations * time per iteration
            prev_time = time.time()
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )
            # Save a set of images from the test set every 100 iterations
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        # Update learning rate
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    ## Save the model after training
    torch.save(G_AB.state_dict(), "save/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
    torch.save(G_BA.state_dict(), "save/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
    torch.save(D_A.state_dict(), "save/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
    torch.save(D_B.state_dict(), "save/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
    print("\nsave my model finished !!")
    #    ## Save the model every few epochs
    #     if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
    #         # Save model checkpoints
    #         torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
    #         torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))


if __name__ == '__main__':
    train()  ## Train the model