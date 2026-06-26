import torch
import torch.nn as nn

#Adversarial Loss
gan_loss = nn.BCEWithLogitsLoss()

#Pixel-wise Recostructon Loss
l1_loss=nn.L1Loss()

