import torch
import torch.nn as nn
from torch.utils.model_zoo import tqdm
import os

def test_anomaly_detection(opt, generator, discriminator, encoder,
                           dataloader, device, kappa=1.0):
    generator.load_state_dict(torch.load("results/generator"))
    discriminator.load_state_dict(torch.load("results/discriminator"))
    encoder.load_state_dict(torch.load("results/encoder"))

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device).eval()
    criterion = nn.MSELoss()

    if not os.path.isfile("results/score.csv"):
      with open("results/score.csv", "a") as f:
        f.write("img_distance,anomaly_score,z_distance\n")

    for img in tqdm(dataloader):

        real_img = img.to(device)

        real_z = encoder(real_img)
        fake_img = generator(real_z)
        fake_z = encoder(fake_img)

        real_feature = discriminator.forward_features(real_img)
        fake_feature = discriminator.forward_features(fake_img)

        # Scores for anomaly detection
        img_distance = criterion(fake_img, real_img)
        loss_feature = criterion(fake_feature, real_feature)
        anomaly_score = img_distance + kappa * loss_feature

        z_distance = criterion(fake_z, real_z)

        with open("results/score.csv", "a") as f:
            f.write(f"{img_distance},"
                    f"{anomaly_score},{z_distance}\n")
