import keras
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

