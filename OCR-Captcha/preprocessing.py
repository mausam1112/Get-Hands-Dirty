import keras
import tensorflow as tf
from configs import Hyperparater as hypr
from keras import layers, ops


def character_mapping(characters):

    # Mapping characters  to integers 
    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    return char_to_num, num_to_char


def split_data(images, labels, train_split=0.9, shuffle=True):
    # calculate size of dataset
    size = len(images)
    # create array of indices and shuffle if required
    indices = ops.arange(size)
    if shuffle:
        indices = keras.random.shuffle(indices)
    
    # calculate size of train set
    train_size = int(size * train_split)

    # split
    x_train, y_train = images[indices[:train_size]], labels[indices[:train_size]]
    x_valid, y_valid = images[indices[train_size:]], labels[indices[train_size:]]

    return x_train, x_valid, y_train, y_valid

def encode_sample(img_path, label, char_to_num):
    # load image
    img = tf.io.read_file(img_path)
    # decode and transform to grayscale with channel 1
    img = tf.io.decode_png(img, channels=1)
    # converting to float32 and normalizing the img by converting pixel value between 0 and 1
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize image
    img = ops.image.resize(img, [hypr.IMG_HEIGHT, hypr.IMG_WIDTH])
    # transpose [h, w, c] to [w, h, c] so that width represents time dimension
    img = ops.transpose(img, axes=[1, 0, 2])
    # map the characters in labels to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # return transformed image and encoded label
    return {"image": img, "label": label}