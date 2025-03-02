import os
import glob
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

#########################################################################
############################  datasets.py  ###################################

## If the input dataset is grayscale images, convert the images to RGB (not needed for the facades dataset used here)
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


## Build the dataset
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False,
                 mode="train"):  ## (root = "./datasets/facades", unaligned=True: unaligned data)
        self.transform = transforms.Compose(transforms_)  ## Transform to tensor data
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))  ## "./datasets/facades/trainA/*.*"
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))  ## "./datasets/facades/trainB/*.*"

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])  ## Take one image from A

        if self.unaligned:  ## If using unaligned data, randomly take one image from B
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # If the image is grayscale, convert it to RGB
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        # Convert RGB images to tensor for easier computation, return dictionary data
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    ## Get the length of data A and B
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))