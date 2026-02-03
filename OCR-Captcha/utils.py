import os
from configs import Paths, Hyperparater
from pathlib import Path


def get_images_labels():
    images = sorted(
        list(map(str, list(Path(Paths.DATA_DIR).glob("*.png"))))
    ) # type: ignore
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
    characters = sorted(
        list(
            set(char for label in labels for char in label)
        )
    )


    print("Number of images: ", len(images))
    print("Number of labels: ", len(labels))
    print(f"Number of unique characters: {len(characters)}")
    return images, labels


if __name__=="__main__":
    images, labels = get_images_labels()
