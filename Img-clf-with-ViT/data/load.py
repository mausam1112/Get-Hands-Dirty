import os
import keras
from config import setup


def filter_corrupted_images():
    dirs_lvl1 = list(
        filter(
            os.path.isdir,
            [
                os.path.join(setup.DATA_PATH, path)
                for path in os.listdir(setup.DATA_PATH)
            ],
        )
    )
    assert len(dirs_lvl1) == 2, "Expected 2 directories of train and test"

    num_skipped = 0

    for dir_lvl1 in dirs_lvl1:
        dirs_lvl2 = list(
            filter(
                os.path.isdir,
                [os.path.join(dir_lvl1, path) for path in os.listdir(dir_lvl1)],
            )
        )
        for dir_lvl2 in dirs_lvl2:
            for fname in os.listdir(dir_lvl2):
                fpath = os.path.join(dir_lvl2, fname)
                # print(fname, os.path.exists(fpath))
                try:
                    fobj = open(fpath, "rb")
                    is_jfif = b"JFIF" in fobj.peek(10)
                finally:
                    fobj.close()

                if not is_jfif:
                    num_skipped += 1
                    # Delete corrupted image
                    os.remove(fpath)
    print(f"Deleted {num_skipped} images.")
