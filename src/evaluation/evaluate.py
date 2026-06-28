import os
import glob
import numpy as np
from PIL import Image
from skimage.metrics import(
    peak_signal_noise_ratio,
    structural_similarity,
)

def load_image(image_path):
    image=Image.open(image_path).convert("RGB")
    image=np.array(image)
    return image

def evaluate(prediction_folder, ground_truth_folder):

    prediction_paths = sorted(
        glob.glob(os.path.join(prediction_folder, "*.png"))+
        glob.glob(os.path.join(prediction_folder, "*.jpg"))+
        glob.glob(os.path.join(prediction_folder, "*.jpeg"))
    )

    ground_truth_paths = sorted(
        glob.glob(os.path.join(ground_truth_folder, "*.png"))+
        glob.glob(os.path.join(ground_truth_folder, "*.jpg"))+
        glob.glob(os.path.join(ground_truth_folder, "*.jpeg"))
    )

    print(f"Found {len(prediction_paths)} predictions")
    print(f"Found {len(ground_truth_paths)} ground truth images")

    if len(prediction_paths) != len(ground_truth_paths):
        raise ValueError(
        "Prediction and ground truth folders contain different numbers of images."
    )

total_psnr = 0.0
total_ssim = 0.0