import os

os.environ["KERAS_BACKEND"] = "tensorflow"

DATA_PATH = "D:/Projects/Hands-Dirty/dataset/clf/cat-dog dataset"
MODEL_PATH = "D:/Projects/Hands-Dirty/Img-clf-with-ViT/saved_model"
PREFIX = "exp"
IMAGE_SIZE = (90, 90)
CHANNEL = 3
BATCH_SIZE = 128

learning_rate = 0.001
weight_decay = 0.0001

num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images

# num_patches = (image_size // patch_size) ** 2
num_patches = (IMAGE_SIZE[0] // patch_size) ** 2

projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    1024,
    512,
]  # Size of the dense layers of the final classifier
