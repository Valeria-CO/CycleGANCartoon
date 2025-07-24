import os
import torch.utils.data as data
from PIL import Image


class CartoonDataset(data.Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.dataX = os.path.join(path, "photo")
        self.dataY = os.path.join(path, "style")

        self.imagesX = sorted(os.listdir(self.dataX))
        self.imagesY = sorted(os.listdir(self.dataY))

    def __len__(self):
        return max(len(self.imagesX), len(self.imagesY))

    def __getitem__(self, index):
        X = Image.open(os.path.join(self.dataX, self.imagesX[index % len(self.imagesX)])).convert("RGB")
        Y = Image.open(os.path.join(self.dataY, self.imagesY[index])).convert()

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        return X, Y

