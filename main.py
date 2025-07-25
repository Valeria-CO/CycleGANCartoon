import os
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

from dataset import CartoonDataset
from model import GenCartoon, DisCartoon



transforms = tfs_v2.Compose([
    tfs_v2.RandomHorizontalFlip(p=0.5),
    tfs_v2.Resize((256, 256)),
    tfs_v2.ToImage(),
    tfs_v2.ToDtype(torch.float32, scale=True),
    tfs_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CartoonDataset(path="dataset", transform=transforms)
data_train = data.DataLoader(dataset, batch_size=1, shuffle=True)

G_X_Y = GenCartoon().to(device)
G_Y_X = GenCartoon().to(device)
D_X = DisCartoon().to(device)
D_Y = DisCartoon().to(device)

G_X_Y.load_state_dict(torch.load("weights/G_X_Y.pth"))
G_Y_X.load_state_dict(torch.load("weights/G_Y_X.pth"))
D_X.load_state_dict(torch.load("weights/D_X.pth"))
D_Y.load_state_dict(torch.load("weights/D_Y.pth"))

adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()

optimizer_G = torch.optim.Adam(list(G_X_Y.parameters()) + list(G_Y_X.parameters()),lr=0.00001, betas=(0.5, 0.999))
optimizer_D_X = optim.Adam(D_X.parameters(), lr=0.00001, betas=(0.5, 0.999))
optimizer_D_Y = optim.Adam(D_Y.parameters(), lr=0.00001, betas=(0.5, 0.999))

optimizer_G.load_state_dict(torch.load("weights/opt_G.pth"))
optimizer_D_X.load_state_dict(torch.load("weights/opt_D_X.pth"))
optimizer_D_Y.load_state_dict(torch.load("weights/opt_D_Y.pth"))


for epoch in range(0):

    G_X_Y.train()
    G_Y_X.train()
    D_X.train()
    D_Y.train()
    loop = tqdm(data_train, desc=f"Epoch {epoch}/{200}", leave=False)
    for real_X, real_Y in loop:
        real_X = real_X.to(device)
        real_Y = real_Y.to(device)

        #  Train Generator

        fake_Y = G_X_Y(real_X)
        fake_X = G_Y_X(real_Y)

        recon_X = G_Y_X(fake_Y)
        recon_Y = G_X_Y(fake_X)

        pred_fake_X = D_X(fake_X)
        pred_fake_Y = D_Y(fake_Y)
        loss_G_Y_X = adversarial_loss(pred_fake_X, torch.ones_like(pred_fake_X))
        loss_G_X_Y = adversarial_loss(pred_fake_Y, torch.ones_like(pred_fake_Y))

        loss_cycle_X = cycle_loss(recon_X, real_X)
        loss_cycle_Y = cycle_loss(recon_Y, real_Y)

        identity_Y = G_X_Y(real_Y)
        identity_X = G_Y_X(real_X)
        loss_identity = cycle_loss(identity_X, real_X) + cycle_loss(identity_Y, real_Y)

        lambda_cycle = 3.0
        lambda_identity = 0.0

        loss_G = (
                loss_G_X_Y +
                loss_G_Y_X +
                lambda_cycle * (loss_cycle_X + loss_cycle_Y) +
                lambda_identity * loss_identity)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        #  Train Discriminator

        noise_std = 0.1
        flip_prob = 0.05

        # Дискриминатор D_Y
        real_Y_noisy = real_Y + torch.randn_like(real_Y) * noise_std
        fake_Y_noisy = fake_Y.detach() + torch.randn_like(fake_Y) * noise_std

        # loss_D_Y_real = adversarial_loss(D_Y(real_Y), torch.ones_like(D_Y(real_Y)))
        real_label = torch.full_like(D_Y(real_Y), 0.9)
        fake_label = torch.zeros_like(D_Y(fake_Y))
        if torch.rand(1).item() < flip_prob:
            real_label, fake_label = fake_label, real_label
        loss_D_Y_real = adversarial_loss(D_Y(real_Y_noisy), real_label)
        loss_D_Y_fake = adversarial_loss(D_Y(fake_Y_noisy), fake_label)
        loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5

        optimizer_D_Y.zero_grad()
        loss_D_Y.backward()
        optimizer_D_Y.step()

        # Дискриминатор D_X
        # loss_D_X_real = adversarial_loss(D_X(real_X), torch.ones_like(D_X(real_X)))
        real_X_noisy = real_X + torch.randn_like(real_X) * noise_std
        fake_X_noisy = fake_X.detach() + torch.randn_like(fake_X) * noise_std
        real_label_X = torch.full_like(D_X(real_X), 0.9)
        fake_label_X = torch.zeros_like(D_X(fake_X))
        if torch.rand(1).item() < flip_prob:
            real_label_X, fake_label_X = fake_label_X, real_label_X

        loss_D_X_real = adversarial_loss(D_X(real_X_noisy), real_label_X)
        loss_D_X_fake = adversarial_loss(D_X(fake_X_noisy), fake_label_X)
        loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5

        optimizer_D_X.zero_grad()
        loss_D_X.backward()
        optimizer_D_X.step()


    torch.save(G_X_Y.state_dict(), "weights/G_X_Y.pth")
    torch.save(G_Y_X.state_dict(), "weights/G_Y_X.pth")
    torch.save(D_X.state_dict(), "weights/D_X.pth")
    torch.save(D_Y.state_dict(), "weights/D_Y.pth")
    torch.save(optimizer_G.state_dict(), "weights/opt_G.pth")
    torch.save(optimizer_D_X.state_dict(), "weights/opt_D_X.pth")
    torch.save(optimizer_D_Y.state_dict(), "weights/opt_D_Y.pth")


    print(f"[Epoch {epoch}] loss_G: X-Y : {loss_G_X_Y.item():.4f}, Y-X: {loss_G_Y_X.item():.4f},loss_identity: {loss_identity.item(): .4f}, "
          f"cycle-X: {loss_cycle_X.item():.4f}, cycle-Y: {loss_cycle_Y.item():.4f}, "
          f"loss_D_X: {loss_D_X.item():.4f}, loss_D_Y: {loss_D_Y.item():.4f}")


    # result
    G_X_Y.eval()

    with torch.no_grad():
        img = Image.open("001.jpg").convert("RGB")
        img_tensor = transforms(img).unsqueeze(0).to(device)
        predict = G_X_Y(img_tensor)

    predict = predict * 0.5 + 0.5 #(predict + 1) / 2
    img_out = predict.squeeze(0).permute(1, 2, 0).cpu().numpy()

    os.makedirs("results", exist_ok=True)

    plt.imsave(f"results/epoch_{epoch}.png", img_out)


# for i in range(1, 21):
#     img = Image.open(f"results/{i}photo.jpg").convert("RGB")
#     img = transforms(img).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         style_img = G_X_Y(img)
#         photo_img = G_Y_X(style_img)
#
#     style_img = (style_img * 0.5 + 0.5).squeeze().permute(1, 2, 0).cpu().numpy()
#     photo_img = (photo_img * 0.5 + 0.5).squeeze().permute(1, 2, 0).cpu().numpy()
#     plt.imsave(f"results/{i}style_img.png", style_img)
#     plt.imsave(f"results/{i}photo_img.png", photo_img)

