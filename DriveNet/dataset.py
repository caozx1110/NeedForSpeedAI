import os
import re
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


label_dict = {'A': 0, 'D': 1, 'N': 2, 'W': 3, 'WA': 4, 'WD': 5}


class nfs_cls_dataset(Dataset):
    """ Used for creating classification data loader """

    train_label_folder = 'train_ann'
    val_label_folder = 'val_ann'
    test_label_folder = 'test_ann'

    def __init__(self, data_path, mode, input_trans):
        super(nfs_cls_dataset, self).__init__()
        assert mode == 'train' or mode == 'test' or mode == 'val', \
            'The input mode of the dataset if wrong'
        if mode == 'train':
            self.data_path = os.path.join(data_path, self.train_label_folder)
        elif mode == 'test':
            self.data_path = os.path.join(data_path, self.test_label_folder)
        else:
            self.data_path = os.path.join(data_path, self.val_label_folder)
        self.transform = input_trans
        self.data = []
        self.label = []
        list_dir = os.listdir(self.data_path)
        print("len", len(list_dir))
        for i in range(len(list_dir)):
            # print("loaded", i)
            lbl_name = list_dir[i]
            key = re.findall(r'[a-zA-Z]+', lbl_name)[0]
            abs_lbl_path = os.path.join(self.data_path, lbl_name)
            img = np.array(Image.open(abs_lbl_path)) / 2
            label = label_dict[key]
            self.data.append(img)
            self.label.append(label)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        # lbl_name = os.listdir(self.data_path)[index]
        # key = re.findall(r'[a-zA-Z]+', lbl_name)[0]
        # abs_lbl_path = os.path.join(self.data_path, lbl_name)
        # lbl_img = np.array(Image.open(abs_lbl_path)) / 2
        img = self.transform(img)
        # output_label = label_dict[key]
        return img, label

    def __len__(self):
        return len(os.listdir(self.data_path))
