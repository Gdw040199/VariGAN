import random
import torch
from torch.autograd import Variable

#########################################################################
############################  utils.py  ###################################

# Buffer for previously generated samples
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):  # Push an image into the buffer and then pop one out
        to_return = []  # Ensure randomness of data to improve discriminator's ability to distinguish real and fake images
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:  # If the buffer is not full, keep adding images
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:  # If the buffer is full, with 50% probability, take an image from the buffer or use the current input image
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


# Set the learning rate as the initial learning rate multiplied by the value of the given lr_lambda function
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):  # (n_epochs = 50, offset = epoch, decay_start_epoch = 30)
        assert (
                           n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"  # Assertion: Ensure n_epochs > decay_start_epoch
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):  # return 1 - max(0, epoch - 30) / (50 - 30)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)