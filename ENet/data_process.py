import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_data(root, img_folder='', label_folder=''):
    root_list = os.listdir(root)
    for item in root_list:
        json_list = os.path.join(root, item)
        name = item[: len(item) - 5]
        img_path = os.path.join(json_list, 'img.png')
        label_path = os.path.join(json_list, 'label.png')
        img = Image.open(img_path)
        label = Image.open(label_path)
        img.save(os.path.join(img_folder, name + '.png'))
        label.save(os.path.join(label_folder, name + '.png'))


def split_dataset(root, train_path, test_path, val_path, test_ratio, val_ratio):
    assert test_ratio + val_ratio < 1, \
        'The sum of ``test_ratio`` and ``val_ratio`` should be less than 1'
    root_ann_path = root + '_ann'
    train_ann_path = train_path + '_ann'
    test_ann_path = test_path + '_ann'
    val_ann_path = val_path + '_ann'
    img_path_set = os.listdir(root)
    order = [i for i in range(len(img_path_set))]
    length = len(order)
    test_order = random.sample(order, round(test_ratio * length))
    for item in test_order:
        order.remove(item)
        name = img_path_set[item][:len(img_path_set[item]) - 4]
        test_img = Image.open(os.path.join(root, img_path_set[item]))
        test_ann = Image.open(os.path.join(root_ann_path, img_path_set[item]))
        test_img.save(os.path.join(test_path, name + '.png'))
        test_ann.save(os.path.join(test_ann_path, name + '.png'))
    val_order = random.sample(order, round(val_ratio * length))
    for item in val_order:
        order.remove(item)
        name = img_path_set[item][:len(img_path_set[item]) - 4]
        val_img = Image.open(os.path.join(root, img_path_set[item]))
        val_ann = Image.open(os.path.join(root_ann_path, img_path_set[item]))
        val_img.save(os.path.join(val_path, name + '.png'))
        val_ann.save(os.path.join(val_ann_path, name + '.png'))
    for item in order:
        name = img_path_set[item][:len(img_path_set[item]) - 4]
        train_img = Image.open(os.path.join(root, img_path_set[item]))
        train_ann = Image.open(os.path.join(root_ann_path, img_path_set[item]))
        train_img.save(os.path.join(train_path, name + '.png'))
        train_ann.save(os.path.join(train_ann_path, name + '.png'))


extract_data('dataset', img_folder='root', label_folder='root_ann')

split_dataset('root', 'nfs/train', 'nfs/test', 'nfs/val', .1, .1)


img = Image.open('nfs/train_ann/A1.png')
plt.imshow(img)
x = np.array(img)
print(np.max(x))
np.bincount(x.flatten(), minlength=3)
