import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():
    print(os.getcwd())
    data_dir = 'Brain_MRI_Images\Train'

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = dset.ImageFolder(data_dir, transform=transform)
    normal_dataset = torch.utils.data.Subset(dataset, range(0, 135))
    normal_dataloader = DataLoader(normal_dataset, batch_size=16, shuffle=True, num_workers=4)

    tumor_dataset = torch.utils.data.Subset(dataset, range(135, 320))
    tumor_dataloader = DataLoader(tumor_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Decide which device we want to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Plot some training images
    real_batch = next(iter(tumor_dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

if __name__ == '__main__':
    main()