import os
import torch
import glob
import argparse
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

def predict_folder(generator, input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_paths=sorted(
        glob.glob(os.path.join(input_folder, "*.png"))+
        glob.glob(os.path.join(input_folder, "*.jpg"))+
        glob.glob(os.path.join(input_folder, "*.jpeg"))
        )

    print(f"Found {len(image_paths)} images")

    for index,image_path in enumerate(image_paths, start=1):
        image=load_image(image_path)
        prediction=predict(generator, image)

        filename=os.path.basename(image_path)
        save_path=os.path.join(output_folder, filename)

        save_image(prediction.cpu(), save_path)
        print(f"[{index}/{len(image_paths)}] Saved: {save_path}")
    
    print("Folder prediction completed!")

def predict(generator, image):
    
    with torch.no_grad():
        prediction=generator(image)
        if isinstance(prediction, (list, tuple)):
            prediction=prediction[0]
    return prediction 

if __name__=="__main__":
    print("Step 1: Main started")
    
    parser=argparse.ArgumentParser(description="Cloud Removal Inference")

    parser.add_argument(
        "--input",
        type=str,
        default="datasets/raw/RICE_DATASET/RICE/RICE1/cloud/1.png",
        help="Path to cloudy input image"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predicted.png",
        help="Path to save prediction"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/generator_epoch_100.pth",
        help="Generator checkpoint"
    )

    args=parser.parse_args()

    checkpoint_path=args.checkpoint
    input_image=args.input
    output_image=args.output


    print("Step 2 Creating output folder")
    os.makedirs("outputs", exist_ok=True)

    print("Step 3: Loading generattor")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        exit()
    generator=load_generator(checkpoint_path)

    print("Step 4: Loading image")
    if not os.path.exists(input_image):
        print(f"Error: Input not found: {input_image}")
        exit()

    if os.path.isdir(input_image):
        print("Folder detected")

        predict_folder(
            generator,
            input_image,
            output_image
        )

    else:
        print("Single image detected")
        image=load_image(input_image)
        prediction=predict(generator, image)
        save_image(prediction.cpu(), output_image)
        print("Finished Successfully")

print("=" * 50)
print("Cloud Removal Inference")
print("=" * 50)
print(f"Checkpoint : {checkpoint_path}")
print(f"Input      : {input_image}")
print(f"Output     : {output_image}")
print(f"Device     : {DEVICE}")
print("=" * 50)
