import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os


# our custom dataset class
class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # if the mask is not grayscale, remove the convert("L") part

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # convert the mask to a tensor
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


# training function
def train_fcn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  # number of classes in the dataset
    batch_size = 4
    num_epochs = 50

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(
        img_dir="path/to/train_images",
        mask_dir="path/to/train_masks",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    model = models.segmentation.fcn_resnet50(pretrained=False)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "custom_fcn.pth")
    print("Training completed! Model saved as custom_fcn.pth")


if __name__ == "__main__":
    train_fcn()
