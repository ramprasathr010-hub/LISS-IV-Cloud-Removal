from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import RICEDataset

def get_dataloader(cloud_dir, label_dir , batch_size=8, shuffle=True):
    transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    dataset = RICEDataset(
        cloud_dir=cloud_dir,
        label_dir=label_dir,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return dataloader