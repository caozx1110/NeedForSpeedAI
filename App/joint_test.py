import os
import random
import time

from PIL import Image
import sys
from torch import optim
import matplotlib.pyplot as plt
from collections import OrderedDict
from export import *
from enet import ENet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def joint_test(seg_model, cls_model, img, class_encoding, DEVICE):
    """
    Perform the total test.
    """
    pre = alter_predict(seg_model, img, DEVICE)
    pre = process_pre(pre, class_encoding)
    result = cls_model(pre)
    result = torch.argmax(result, 1).cpu().item()
    return result


class_encoding = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('road', (255, 0, 0)),
    ('car', (0, 255, 0)),
])


if __name__ == '__main__':
    model = ENet(num_classes=3).to(DEVICE)
    checkpoint = torch.load('./save/aug0/nfs_enet', map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['state_dict'])
    ClassModel = torch.load("./save/drive.pth", map_location=torch.device(DEVICE))
    """"""
    import matplotlib.image
    folder = '../Data/Collect/data_raw/lph1/'
    list_dir = os.listdir(folder)
    random.shuffle(list_dir)
    for f in list_dir:
        img = Image.open(os.path.join(folder, f))
        pre = alter_predict(model, img, DEVICE)
        render = pre2render(pre, class_encoding)
        matplotlib.image.imsave(os.path.join('../Data/drive/', f), render)
        time.sleep(0.5)
    """"""
    # test_img = Image.open('./temp.png')
    #
    # st = time.time()
    # pre = alter_predict(model, test_img, DEVICE)
    # print('seg', time.time() - st)
    #
    # render = pre2render(pre, class_encoding)
    #
    # plt.imshow(render)
    # plt.show()
