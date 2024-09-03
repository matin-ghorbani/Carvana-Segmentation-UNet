import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.imgs = os.listdir(img_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.imgs[idx].replace('.jpg', '_mask.gif'))

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        mask[mask == 255.] = 1.
        if self.transform is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations['image']
            mask = augmentations['mask']

        return img, mask
