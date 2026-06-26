import os
from PIL import Image
from torch.utils.data import Dataset

class RICEDataset(Dataset):
    def __init__(self, cloud_dirs, label_dirs, transform=None):
        self.transform=transform
        self.samples=[]

        for cloud_dir, label_dir in zip(cloud_dirs, label_dirs):
            image_names=sorted(os.listdir(cloud_dir))

            for image_name in image_names:
                self.samples.append((
                    os.path.join(cloud_dir, image_name),
                    os.path.join(label_dir,image_name)  
                ))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cloud_path, label_path = self.samples[idx]
        
        cloud_image=Image.open(cloud_path).convert("RGB")
        label_image=Image.open(label_path).convert("RGB")

        if self.transform:
            cloud_image=self.transform(cloud_image)
            label_image=self.transform(label_image)
        
        return cloud_image,label_image