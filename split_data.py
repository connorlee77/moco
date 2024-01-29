import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

data_path = '/media/hdd2/data/thermal_ssl/'

file_paths = glob.glob(os.path.join(data_path, '*', '*'))
min_w, min_h = 1024, 1024
for path in file_paths:
    img = Image.open(path)
    w, h = img.size
    min_w = min(min_w, w)
    min_h = min(min_h, h)
print(min_w, min_h)

# train_paths, test_paths = train_test_split(file_paths, test_size=0.1, shuffle=True)

# def copy_files(paths, dest):
#     for path in paths:
#         cls, name = path.split(os.path.sep)[-2:]
#         shutil.copy(path, os.path.join(dest, cls, name))

# copy_files(train_paths, os.path.join(data_path, 'train'))
# copy_files(test_paths, os.path.join(data_path, 'val'))