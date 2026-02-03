from enum import Enum, StrEnum

class Hyperparater(Enum):
    BATCH_SIZE = 16     # Batch size for training and validation

    # Desired image dimensions
    IMG_WIDTH = 200
    IMG_HEIGHT = 50


class Paths(StrEnum):
    DATA_DIR = "../dataset/ocr-captcha/captcha_images_v2"