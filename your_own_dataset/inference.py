import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from data_loader import TrainDataModule, get_all_test_dataloaders

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mvtec_ad.model import Generator, Discriminator, Encoder
from torch import Tensor


class Fanogan(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        # Image size is 64 x 64, latent dim should be 4 x 4 x 128
        
        self.generator = Generator(opt)
        self.discriminator = Discriminator(opt)
        self.encoder = Encoder(opt)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.generator(x)
        return x

    def detect_anomaly(self, x: Tensor):
        rec = self(x)
        anomaly_map = torch.abs(x - rec)
        
        real_z = self.encoder(x)
        fake_img = self.generator(real_z)
        fake_z = self.encoder(fake_img)

        real_feature = self.discriminator.forward_features(x)
        fake_feature = self.discriminator.forward_features(fake_img)

        # Scores for anomaly detection
        kappa=1.0
        img_distance = self.loss_fn(fake_img, x)
        loss_feature = self.loss_fn(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature



        
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }



def main(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataloaders = get_all_test_dataloaders(
    split_dir="/content/f-AnoGAN/your_own_dataset/data/splits",
    target_size=[128,128],
    batch_size=32)
    model=Fanogan(opt)

    model.generator.load_state_dict(torch.load("results/generator"))
    model.discriminator.load_state_dict(torch.load("results/discriminator"))
    model.encoder.load_state_dict(torch.load("results/encoder"))
    
    from evaluate import Evaluator
    evaluator = Evaluator(model, model.device, test_dataloaders)
    metrics, fig_metrics, fig_example = evaluator.evaluate()



    
    

    
    

    

    


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
    parser.add_argument("--n_grid_lines", type=int, default=10,
                        help="number of grid lines in the saved image")
    parser.add_argument("--force_download", "-f", action="store_true",
                        help="flag of force download")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128,
                        help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1,
                        help="number of image channels (If set to 1, convert image to grayscale)")
    parser.add_argument("--n_iters", type=int, default=None,
                        help="value of stopping iterations")
    opt = parser.parse_args()

    main(opt)


    