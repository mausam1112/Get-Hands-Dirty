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


def gen_train_val_data():
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(setup.DATA_PATH, "training_set"),
        labels="inferred",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=setup.IMAGE_SIZE,
        batch_size=setup.BATCH_SIZE,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(setup.DATA_PATH, "test_set"),
        labels="inferred",
        image_size=setup.IMAGE_SIZE,
        batch_size=setup.BATCH_SIZE,
    )

    return train_ds, val_ds, test_ds
