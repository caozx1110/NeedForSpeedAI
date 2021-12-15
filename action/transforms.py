import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision.transforms import ToPILImage


class PILToLongTensor:
    """
    Convert a ``PIL Image`` to a ``torch.LongTensor``.
    """

    def __call__(self, img):
        """
        Perform the conversion from a ``PIL Image`` to a ``torch.LongTensor``.
        Return a ``LongTensor`` object.
        """
        if not isinstance(img, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(img)))

        # handle numpy array
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))

        # Reshape tensor
        chn = len(img.mode)
        img = img.view(img.size[1], img.size[0], chn)

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0, 2).contiguous().long().squeeze_()


class LongTensorToRGBPIL(object):
    """
    Convert a ``torch.LongTensor`` to a ``PIL image``.
    The input is a ``torch.LongTensor`` where each pixel's value identifies the class.
    The encoding value should be a ordered dictionary to provide label, class name and color
    """
    def __init__(self, rgb_encoding):
        self.rgb_encoding = rgb_encoding

    def __call__(self, tensor):
        """
        Perform the conversion from ``torch.LongTensor`` to a ``PIL image``
        """
        # Check if label_tensor is a LongTensor
        if not isinstance(tensor, torch.LongTensor):
            raise TypeError("label_tensor should be torch.LongTensor. Got {}".format(type(tensor)))
        # Check if encoding is a ordered dictionary
        if not isinstance(self.rgb_encoding, OrderedDict):
            raise TypeError("encoding should be an OrderedDict. Got {}".format(type(self.rgb_encoding)))

        # label_tensor might be an image without a channel dimension, in this case unsqueeze it
        if len(tensor.size()) == 2:
            tensor.unsqueeze_(0)
        color_tensor = torch.ByteTensor(3, tensor.size(1), tensor.size(2))

        for index, (class_name, color) in enumerate(self.rgb_encoding.items()):
            # Get a mask of elements equal to index
            mask = torch.eq(tensor, index).squeeze_()
            # Fill color_tensor with corresponding colors
            for channel, color_value in enumerate(color):
                color_tensor[channel].masked_fill_(mask, color_value)
        return ToPILImage()(color_tensor)
