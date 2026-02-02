from enum import Enum
from pathlib import Path


class Paths(Enum):
    DATA_DIR = Path("../dataset/ocr-captcha/captcha_images_v2")