import os
from PIL import Image
from torch.utils.data import Dataset

class RICEDatasets(Dataset):
    def __init__(self, cloud_dir, label_dir, transform=None):
        self.cloud_dir=cloud_dir
        self.label_dir=label_dir
        self.transform=transform

        self.image_name=sorted(os.listdir(cloud_dir))

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name=self.image_names[idx]

        cloud_path=os.path.join(self.cloud_dir,image_name)
        label_path=os.path.join(self.label_dir,image_name)

        cloud_image=Image.open(cloud_path).convert("RGB")
        label_image=Image.open(label_path).convert("RGB")

        if self.transform:
            cloud_image=self.transform(cloud_image)
            label_image=self.transform(label_image)