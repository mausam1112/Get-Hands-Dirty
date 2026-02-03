from keras import layers


def character_mapping(characters):

    # Mapping characters  to integers 
    char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

    # Mapping integers back to original characters
    num_to_char = layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
    )

    return char_to_num, num_to_char
