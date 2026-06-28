import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from src.models.u2net import U2NET
from src.config.config import DEVICE, IMAGE_SIZE

def load_generator(checkpoint_path):
    generator=U2NET(in_ch=3,out_ch=3).to(DEVICE)

    generator.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )
    generator.eval()
    return generator

transform =transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image=Image.open(image_path).convert("RGB")
    image=transform(image)
    image=image.unsqueeze(0)
    image=image.to(DEVICE)
    return image

def predict(generator, image):
    
    with torch.no_grad():
        prediction=generator(image)
        if isinstance(prediction, (list, tuple)):
            prediction=prediction[0]
    return prediction 

if __name__=="__main__":
    print("Step 1: Main started")
    checkpoint_path="checkpoints/generator_epoch_100.pth"
    input_image="datasets/raw/RICE_DATASET/RICE/RICE1/cloud/1.png"
    output_image="outputs/predicted.png"

    print("Step 2 Creating output folder")
    os.makedirs("outputs", exist_ok=True)

    print("Step 3: Loading generattor")
    generator=load_generator(checkpoint_path)

    print("Step 4: Loading image")
    image=load_image(input_image)

    print("Step 5: Running prediction")
    prediction=predict(generator, image)

    print("Step 6: Saving image")
    save_image(prediction.cpu(), output_image)
    print("Finished Successfully")
