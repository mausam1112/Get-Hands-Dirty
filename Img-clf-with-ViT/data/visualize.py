import matplotlib.pyplot as plt
import numpy as np
from keras import ops
from nn.layers.patch import Patches
from config import setup


def display_samples(ds):
    plt.figure(figsize=(8, 4))

    for images, labels in ds.take(1):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(labels[i].numpy())
            plt.axis("off")
        break
    plt.show()


def display_patches(ds):
    data = next(ds.as_numpy_iterator())
    images, target = data  # tuple of images (b, h, w, c) and target
    # print(images.shape, target.shape)
    image = images[np.random.choice(np.arange(images.shape[0]))]

    plt.figure(figsize=(4, 4))
    plt.imshow(image.astype("uint8"))
    plt.axis("off")
    plt.title("Sample image")
    plt.show()

    image_tensor = ops.convert_to_tensor([image])
    patches = Patches(setup.patch_size)(image_tensor)

    print(f"Image size: {setup.IMAGE_SIZE[0]} X {setup.IMAGE_SIZE[1]}")
    print(f"Patch size: {setup.patch_size} X {setup.patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")
    print(f"{patches.shape=}")

    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(4, 4))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = ops.reshape(patch, (setup.patch_size, setup.patch_size, 3))
        plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
        plt.axis("off")
    plt.show()
