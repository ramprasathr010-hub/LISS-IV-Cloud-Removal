import os
import torch
from torch.utils.data import DataLoader
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
            "datasets/raw/RICE_DATASET/RICE/RICE1/cloud",
            "datasets/raw/RICE_DATASET/RICE/RICE2/cloud",
        ],

        
        label_dirs=[
            "datasets/raw/RICE_DATASET/RICE/RICE1/label",
            "datasets/raw/RICE_DATASET/RICE/RICE2/label",
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
    
    print("Dataset Size:", len(train_dataset))
    print("Number of batches:", len(train_loader))


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
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            fake_pair=torch.cat([cloud_img,fake_img], dim=1)
            pred_fake=discriminator(fake_pair)
            target_real=torch.ones_like(pred_fake)

            g_gan_loss=gan_loss(pred_fake, target_real)
            g_l1_loss=l1_loss(fake_img, label_img)
            g_loss=g_gan_loss+LAMBDA_L1*g_l1_loss
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            print(
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f} "
                f"L1: {g_l1_loss.item():.4f} "
            )

        torch.save(generator.state_dict(), os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch+1}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    main()