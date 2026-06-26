import torch

#Dataset<--
IMAGE_SIZE=256
BATCH_SIZE=1
NUM_WORKERS=2

#Training
EPOCHS=100
LEARNING_RATE=2e-4
LAMBDA_L1=100

#Device used
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

#Save paths
CHECKPOINT_DIR="checkpoints"
SAMPLE_DIR="outputs"