import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.config.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LEARNING_RATE,
    LAMBDA_L1,
    DEVICE,
    CHECKPOINT_DIR,
    SAMPLE_DIR,
)

from src.data.dataset import RICEDataset
from src.data.transforms import train_transform

from src.models.u2net import U2NET
from src.models.networks import NLayerDiscriminator

from src.training.loss import gan_loss,l1_loss


def main():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    train_dataset=RICEDataset(
        cloud_dirs=[
            "datasets/raw/RICEDATASET/RICE/RICE1/cloud",
            "datasets/raw/RICEDATASET/RICE/RICE2/cloud",
        ],

        
        label_dirs=[
            "datasets/raw/RICEDATASET/RICE/RICE1/label",
            "datasets/raw/RICEDATASET/RICE/RICE2/label",
        ],
        transform=train_transform,
    ) 

    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    generator=U2NET(in_ch=3, out_ch=3).to(DEVICE)
    
    discriminator=NLayerDiscriminator(
        input_nc=6,
        ndf=64,
        n_layers=3,
    ).to(DEVICE)

    g_optimizer=torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    d_optimizer=torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    g_scaler=GradScaler()
    d_scaler=GradScaler()

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        generator.train()
        discriminator.train()

        for batch_idx, (cloud_img, label_img) in enumerate(train_loader):
            cloud_img=cloud_img.to(DEVICE)
            label_img=label_img.to(DEVICE)
            
            outputs=generator(cloud_img)
            fake_img=outputs[0]

            real_pair=torch.cat([cloud_img,label_img],dim=1)
            fake_pair=torch.cat([cloud_img,fake_img.detach()],dim=1)

            real_pred=discriminator(real_pair)
            fake_pred=discriminator(fake_pair)

            real_target=torch.ones_like(real_pred)
            fake_target=torch.zeros_like(fake_pred)

            real_loss=gan_loss(real_pred, real_target)
            fake_loss=gan_loss(fake_pred, fake_target)
            d_loss=(real_loss+fake_loss)*0.5