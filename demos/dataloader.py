import os
from torch.utils.data import Dataset
import pandas as pd
from torchvision.io import read_image

class ImageDataset(Dataset):
    def __init__(self, img_dir, style, transform=None):
        self.style = style
        self.img_dir = img_dir
        self.images = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.style
