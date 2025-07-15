import os
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
from PIL import Image


class CartoonDataset(data.Dataset):
    def __init__(self, path, transform=None, mode='train'):
        self.transform = transform
        self.dataX = os.path.join(path, 'photo')
        self.dataY = os.path.join(path, 'style')

        self.imagesX = sorted(os.listdir(self.dataX))
        self.imagesY = sorted(os.listdir(self.dataY))

    def __len__(self):
        return max(len(self.imagesX), len(self.imagesY))

    def __getitem__(self, index):
        X = Image.open(os.path.join(self.dataX, self.imagesX[index % len(self.imagesX)])).convert('RGB')
        Y = Image.open(os.path.join(self.dataY, self.imagesY[index])).convert()

        if self.transform:
            X = self.transform(X)
            Y = self.transform(Y)

        return X, Y


# transforms = tfs_v2.Compose([
#     tfs_v2.Resize((256, 256)),
#     tfs_v2.ToImage(),
#     tfs_v2.ToDtype(torch.float32, scale=True),  # [0,255] → [0,1]
#     tfs_v2.Normalize([0.5]*3, [0.5]*3),
# ])
#
#
# dataset = CartoonDataset(path="dataset", transform=transforms, mode="train")
# loader = data.DataLoader(dataset, batch_size=4, shuffle=True)

