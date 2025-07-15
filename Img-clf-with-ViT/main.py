from data import load, visualize
from nn.layers import patch
from config import setup
from nn import model
from utils import utils


def main():
    load.filter_corrupted_images()
    train_ds, val_ds, test_ds = load.gen_train_val_data()
    visualize.display_samples(train_ds)
    # list(train_ds.as_numpy_iterator())    # convert EagerTensor into numpy array

    for idx, ele in enumerate(train_ds.as_numpy_iterator()):
        # print(len(ele), 'image->', ele[0].shape, 'label->', ele[1].shape)
        lyr_patch = patch.Patches(setup.patch_size)(ele[0])
        break

    visualize.display_patches(train_ds)
    img_clf = model.vit_classifier(num_class=1)
    print(img_clf.summary())

    new_suffix = utils.suffix_counter(setup.MODEL_PATH, is_dir=True, prefix=setup.PREFIX)
    print(f"{new_suffix = }")


if __name__ == "__main__":
    main()
