from data import load


def main():
    load.filter_corrupted_images()
    train_ds, val_ds, test_ds = load.gen_train_val_data()


if __name__ == "__main__":
    main()
