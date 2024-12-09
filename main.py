from ViT import ViT
from MNIST_preprocess import MNIST_preprocess_data
import torch
from torch.utils.data import DataLoader
from train import train_model
import random
import numpy as np

RANDOM_SEED = 42
BATCH_SIZE = 2 # number of samples propogated
EPOCHS = 40 # number of iterations of training dataset

LEARNING_RATE = 1e-4
NUM_CLASSES = 10
PATCH_SIZE = 4
IMG_SIZE = 28
INPUT_CHANNELS = 1
NUM_HEADS = 8 # DECIDES HOW MANY ATTENTION HEADS WE WILL USE
DROPOUT = 0.001
HIDDEN_DIMENSION = 768 # HIDDEN DIMENSION OF MLP HEAD
ADAM_WEIGHT_DECAY = 0 # WEIGHT DECAY WE WILL GIVE TO THE OPTIMIZER, IN THE PAPER THE VALUE DOES NOT WORK AS WELL
ADAM_BETAS = (0.9, 0.999)
ACTIVATION = "gelu"
NUMBER_ENCODERS = 4
EMBEDDED_DIMENSION = (PATCH_SIZE ** 2) * INPUT_CHANNELS # 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2 # 49

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# these will have the paths to MNIST dataset, this line will change accordingly based on the dataset
train_dataset, val_dataset, test_dataset = MNIST_preprocess_data("path/to/MNIST_train.csv", "path/to/MNIST_test.csv", RANDOM_SEED)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# init the model
model = ViT(NUM_PATCHES,IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMBEDDED_DIMENSION, NUMBER_ENCODERS, NUM_HEADS, HIDDEN_DIMENSION, DROPOUT, ACTIVATION, INPUT_CHANNELS)

# train the model
train_model(model, train_dataloader, val_dataloader, device, ADAM_BETAS, LEARNING_RATE, ADAM_WEIGHT_DECAY, EPOCHS)
