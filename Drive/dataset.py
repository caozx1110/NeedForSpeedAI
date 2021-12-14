import os
import re
from PIL import Image
from torch.utils.data import Dataset


label_dict = {}


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

    def __getitem__(self, index):
        img_name = os.listdir(self.data_path)[index]
        key = re.findall(r'[a-zA-Z]+', img_name)[0]
        abs_img_path = os.path.join(self.data_path, img_name)
        output_data = self.transform(Image.open(abs_img_path))
        output_label = label_dict[key]
        return output_data, output_label

    def __len__(self):
        return len(os.listdir(self.data_path))
