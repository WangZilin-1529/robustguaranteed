import pandas as pd
import scipy.stats as st
import math
import os
import shutil

ROOT = './advbench/data/'
def calculate_sample_size(proportion, MoE, confi_level):
    p1 = proportion*(1-proportion)
    z = st.norm.ppf(confi_level)
    p2 = (z/MoE)**2
    return math.ceil(p1*p2)

train_root = os.path.join(ROOT, 'tiny-imagenet-200/train')
test_root = os.path.join(ROOT, 'tiny-imagenet-200/val/images')
target_train_root = os.path.join(ROOT, 'tiny-imagenet/train')
target_test_root = os.path.join(ROOT, 'tiny-imagenet/test')
def allocate_train_tinyimagenet():
    train_root = os.path.join(ROOT, 'tiny-imagenet-200/train')
    target_root = os.path.join(ROOT, 'tiny-imagenet/train')
    for (path, dir, file) in os.walk(train_root):
        if 'images' not in path:
            label = path.split('\\')[-1]
            os.mkdir(os.path.join(target_root, label))
    for (path, dir, file) in os.walk(train_root):
        if 'images' in path:
            label = path.split('\\')[-2]
            for img in file:
                shutil.copy(os.path.join(path, img), os.path.join(target_root, label))

def allocate_test_tinyimagenet():
    root_file = './advbench/data/tiny-imagenet-200/val/val_annotations.txt'
    data = pd.read_csv(root_file, header=None)
    print(data)

root_file = './advbench/data/tiny-imagenet-200/val/val_annotations.txt'
data = pd.read_csv(root_file, sep='\t', header=None)
# for (path, dir, file) in os.walk(train_root):
#     if len(file)==0:
#             continue
#     if 'images' not in path:
#         label = path.split('\\')[-1]
#         os.mkdir(os.path.join(target_test_root, label))
for index, row in data.iterrows():
    shutil.copy(os.path.join(test_root, row[0]), os.path.join(target_test_root, row[1], row[0]))