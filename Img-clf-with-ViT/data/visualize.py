import matplotlib.pyplot as plt
import numpy as np


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
