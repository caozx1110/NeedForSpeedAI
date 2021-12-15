import os
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
from collections import OrderedDict

from action.export import *
from action.enet import ENet

os.chdir('./action')


def joint_test(seg_model, cls_model, img, class_encoding, device):
    """
    Perform the total test.
    """
    pre = alter_predict(seg_model, img, device)
    pre = process_pre(pre, class_encoding)
    result = cls_model(pre)
    result = torch.argmax(result, 1).cpu().item()
    return result


device = 'cuda'

class_encoding = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('road', (128, 64, 128)),
    ('car', (64, 0, 128)),
])

""" Test """
num_classes = len(class_encoding)
model = ENet(num_classes).to(device)
optimizer = optim.Adam(model.parameters())
model = utils.load_checkpoint(model, optimizer, './save', 'nfs_enet')[0]
test_img = Image.open('./data/nfs/test/A5.png')
pre = alter_predict(model, test_img, device)
render = pre2render(pre, class_encoding)
plt.imshow(render)
