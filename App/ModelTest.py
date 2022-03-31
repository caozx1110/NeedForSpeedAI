"""
@function: 1. 对ENet模型进行测试; 2. 使用ENet模型进行DriveNet的数据集生成
"""
import os
import random
import time
import torchvision.transforms.functional
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from export import *
from ENet import ENet

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


def rawdata2dataset():
    """
    use the seg ENet model to seg the raw data to form the dataset used by the DriveNet
    """
    model = ENet(num_classes=3).to(DEVICE)
    checkpoint = torch.load('./save/aug2/nfs_enet', map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['state_dict'])

    """"""
    print("start ann...")
    folder = '../Data/Collect/data_raw/lph2/'
    list_dir = os.listdir(folder)
    random.shuffle(list_dir)
    print(len(list_dir))
    for i, f in enumerate(list_dir):
        print(i)
        img = Image.open(os.path.join(folder, f))
        pre = alter_predict(model, img, DEVICE)
        pil_img = torchvision.transforms.functional.to_pil_image(pre.type(torch.uint8))
        # train: test: val = 0.85: 0.1: 0.05
        if i <= 0.85 * len(list_dir):
            path = os.path.join('../Data/drive/train_ann', f)
        elif 0.85 * len(list_dir) < i < 0.95 * len(list_dir):
            path = os.path.join('../Data/drive/test_ann', f)
        else:
            path = os.path.join('../Data/drive/val_ann', f)

        pil_img.save(path)
        # TODO: if you run this in CPU, you'd better add time.sleep, otherwise your CPU may burn (lol
        # time.sleep(0.5)
    print("over")
    """"""


if __name__ == '__main__':
    model = ENet(num_classes=3).to(DEVICE)
    checkpoint = torch.load('./save/aug2/nfs_enet', map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['state_dict'])
    ClassModel = torch.load("./save/drive.pth", map_location=torch.device(DEVICE))

    # test img file name
    test_img = Image.open('../Data/Collect/data_raw/lph2/W1614.jpg')
    
    st = time.time()
    pre = alter_predict(model, test_img, DEVICE)
    print('seg', time.time() - st)
    
    render = pre2render(pre, class_encoding)
    plt.imshow(render)
    import matplotlib
    matplotlib.image.imsave('1.png', render)
    plt.show()
