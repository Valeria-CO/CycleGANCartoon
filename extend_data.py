import torchvision.transforms.v2 as tfs_v2
from PIL import Image
import matplotlib.pyplot as plt


transforms = tfs_v2.RandomHorizontalFlip(p=1)


for e in range(1, 211):
    img = Image.open(f'dataset/photo/{e}.jpg')
    new_img = transforms(img)
    plt.imsave(f'dataset/photo/{220+e}.jpg', new_img)