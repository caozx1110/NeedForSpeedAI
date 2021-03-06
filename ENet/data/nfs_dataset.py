import os
import random
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from . import utils


class nfs_seg_dataset(Dataset):
    """
    Create the dataset of 'Need For Speed', in types of "train", "val" or "test"
    Get the image pair including the original image and the label in the corresponding folder
    """
    train_folder = 'train'
    train_label_folder = 'train_ann'
    val_folder = 'val'
    val_label_folder = 'val_ann'
    test_folder = 'test'
    test_label_folder = 'test_ann'

    # Images extension
    img_extension = '.png'

    color_encoding = OrderedDict([
        ('unlabeled', (0, 0, 0)),
        ('road', (128, 64, 128)),
        ('car', (64, 0, 128)),
    ])

    def __init__(self, root_dir, mode='train', transform=None,
                 label_transform=None, intensity=.5, loader=utils.pil_loader, dual_trans=None):
        super(nfs_seg_dataset, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.intensity = intensity
        self.loader = loader
        self.dual_trans = dual_trans

        if self.mode.lower() == 'train':
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)
            self.train_labels = utils.get_files(os.path.join(
                root_dir, self.train_label_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)
            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_label_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)
            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_label_folder),
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Get the index of the couple of images (original image and the label)
        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)
        if self.mode == 'train' and self.dual_trans is not None:
            # img, label = self.dual_augment(img, label)
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            img = self.dual_trans(img)
            torch.manual_seed(seed)
            label = self.dual_trans(label)
        if self.transform is not None:
            img = self.transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return img, label

    def __len__(self):
        """ Return the length of the dataset. """
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def dual_augment(self, img, lbl):
        if random.random() > self.intensity:
            angle = random.randint(-45, 45)
            img = F.rotate(img, angle)
            lbl = F.rotate(lbl, angle)
        if random.random() > self.intensity:
            img = F.hflip(img)
            lbl = F.hflip(lbl)
        if random.random() > self.intensity:
            img = F.vflip(img)
            lbl = F.vflip(lbl)
        return img, lbl
