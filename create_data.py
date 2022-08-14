import os
import shutil
import random


def smaller_dataset(n):
    current = os.path.join("./data", "Places365/train")
    new = os.path.join("./data", "Places365/train_custom_365_2000")
    dir_list = os.listdir(current)
    for dir_name in dir_list:
        os.makedirs(os.path.join(new, dir_name))
        dir_path = os.path.join(current, dir_name)
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        selected = random.sample(range(0, len(files)), n)
        for file_idx in selected:
            dst = os.path.join(new, "{}/{}".format(dir_name, files[file_idx]))
            shutil.copyfile(os.path.join(dir_path, files[file_idx]), dst)


smaller_dataset(2000)