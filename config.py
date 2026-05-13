"""
Global configuration for conditional GANs benchmark.
"""
import torch

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture
IMG_SIZE = 64
LATENT_DIM = 128
EMBED_DIM = 64
NGF = 64
NDF = 64

# Dataset
TOP_K = None
DATA_DIR = './data'

# Training
BATCH_SIZE = 64
N_EPOCHS = 100
LR_G = 1e-4
LR_D = 1e-4
N_CRITIC = 5
GP_LAMBDA = 10
SAVE_EVERY = 20

# Evaluation
N_EVAL_IMG = 5000
MAX_INCEPTION_IMGS = 2000

# Output directories
OUTPUT_DIR = './outputs'
CHECKPOINT_DIR = './outputs/checkpoints'
GENERATED_DIR = './outputs/generated_images'
RESULTS_DIR = './outputs/results'

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
import numpy as np
np.random.seed(SEED)
