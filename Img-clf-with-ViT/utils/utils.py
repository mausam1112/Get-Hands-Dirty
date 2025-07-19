import os
import matplotlib.pyplot as plt


def plot_history(history, item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

def suffix_counter(path: str, is_dir: bool, prefix: str) -> None | int:
    """
    ARGs:
        path: str
            path where prefix is needed to count.
        is_dir: bool
            If true, count the prefix of the directories only otherwise for files only.
        prefix:
            the starting sub-string to match. It must be separated from the counter with "_"
    Returns:
        None or int
    """
    if not os.path.exists(path):
        print(f"Given {path=} doesn't exists.")
        return None

    is_dir_or_file = os.path.isdir if is_dir else os.path.isfile

    sub_paths = os.listdir(path)

    suffix_champ = 0
    for sub_path in sub_paths:
        full_path = os.path.join(path, sub_path)
        if is_dir_or_file(full_path):
            suffix = sub_path.split("_")[-1]

            if not is_dir:
                suffix = suffix.split(".")[0].strip()
            
            if not suffix.isnumeric():
                continue 
            
            if suffix_champ < (suffix_challenger := int(suffix)):
                suffix_champ = int(suffix_challenger)
    return suffix_champ + 1

