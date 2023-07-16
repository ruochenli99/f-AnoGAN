import os
import sys
from data_loader import TrainDataModule, get_all_test_dataloaders
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from fanogan.test_anomaly_detection import test_anomaly_detection


def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataloader = get_all_test_dataloaders(
    split_dir="/content/f-AnoGAN/your_own_dataset/data/splits",
    target_size=[128,128],
    batch_size=32)
    print(test_dataloader) 
    

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from mvtec_ad.model import Generator, Discriminator, Encoder

    generator = Generator(opt)
    discriminator = Discriminator(opt)
    encoder = Encoder(opt)
    for key,value in test_dataloader.items():
      test_anomaly_detection(opt, generator, discriminator, encoder,
                           value, device)
    #test_anomaly_detection(opt, generator, discriminator, encoder,test_dataloader['absent_septum'], device)


"""
The code below is:
Copyright (c) 2018 Erik Linder-Norén
Licensed under MIT
(https://github.com/eriklindernoren/PyTorch-GAN/blob/master/LICENSE)
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_root", type=str,
                        help="root name of your dataset in test mode")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    opt = parser.parse_args()

    main(opt)
