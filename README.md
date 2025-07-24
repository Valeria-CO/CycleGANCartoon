CycleGAN (Cycle Generative Adversarial Network) is unpaired Image-to-Image translation. 
Traditional GANs need paired data means each input image must have a matching output image. 
But finding such paired images is difficult. CycleGAN can change images from one style to another without needing matching pairs.
It understands the features of the new style and transforms the original images accordingly. 
This makes it useful for tasks like changing seasons in photos, turning one animal into another or converting pictures into paintings.
Architecture as:

CycleGAN has two generators G_X_Y and G_Y_X:
- G_X_Y creates new images in the target style
- G_Y_X transforms generated image back

CycleGAN has two discriminators D_X and D_Y whom decide if images are real or fake 
- D_X distinguishes between real images(x) and generated images G_Y_X(y)
- D_Y distinguishes between real images(y) and generated images G_X_Y(x)

Training CycleGANs can be highly unstable. I experimented with different techniques to stabilize training, including:
- Noise injection and label smoothing for discriminators
- Flip augmentations
- Learning rate tuning and schedulers
- Varying loss weights for cycle and identity loss

I didn't get exactly what I expected, but it's my first project from scratch and I got good experience,
I think I will come back to this a little later. My dataset was quite small (1000 style images and 250 photos), so I think
with the expansion of the dataset the result will get better.
I'll include some examples my results below. These results have been reached 140 epochs of training.
The model version provided here is the best among those trained, and demonstrates key ideas of the CycleGAN architecture.

![  INPUT       /   GENERATED       /  RE-CONVERTED    ](results/00_collage.jpg)
