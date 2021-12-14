import torch
import torchvision
import numpy as np
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import OrderedDict

import utils
from models.enet import ENet
import transforms as ext_transforms


def alter_predict(model, raw_img):
    """
    This API is designed to directly predicted the result.
    ``raw_img`` is the input image in type 'PIL' or nd-array

    The shape of prediction directly from the model will be (N, K, H, W) as the output from
        CrossEntropy loss will contain all the label class. So dimensional reduction is unavoidable.

    Return 2 images:
        the direct predicted image, whose pixel value has a range of 0, K-1
        the encoded image, whose segmented parts has the corresponding encoding color.
    """
    img = torch.unsqueeze(transforms.ToTensor()(raw_img), dim=0).to(device)
    model.eval()
    with torch.no_grad():
        pre = model(img)
    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it by get the index corresponding with the maximum in dim 1.
    # Note the output of the tensor has 3 dimensions and the first dimension represents the number of the image.
    return torch.argmax(pre, 1).cpu()


def pre2render(pre, class_encoding):
    """
    Convert the input prediction with shape (N, H, W) into the nd-array image (N, 3, H, W)
    '3' represents the RGB channels.
    """
    label2rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    render = utils.batch_transform(pre, label2rgb)
    render = torchvision.utils.make_grid(render).numpy()
    return np.transpose(render, (1, 2, 0))


def process_pre(pre, class_encoding):
    """
    Convert prediction into 4 dimensions: N, C, H, W and the pixel range is 0, .5, 1.
    This function is used for create the input of the next network.
    """
    reg_num = len(class_encoding) - 1
    pre = torch.true_divide(torch.FloatTensor(torch.unsqueeze(pre, dim=0)), reg_num)
    return pre


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
pre = alter_predict(model, test_img)
render = pre2render(pre, class_encoding)
plt.imshow(render)
