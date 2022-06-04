import shutil
import os

DATA_ROOT = "data"
RENAME_PREFIX = "2022"
MAX_NAME_BIT = 4
for i, image in enumerate(os.listdir(DATA_ROOT)):
    index_str = str(i)
    supp_space = "0" * (MAX_NAME_BIT - len(index_str))
    target_name = RENAME_PREFIX + supp_space + index_str + ".png"
    shutil.move(os.path.join(DATA_ROOT, image), os.path.join(DATA_ROOT, target_name))