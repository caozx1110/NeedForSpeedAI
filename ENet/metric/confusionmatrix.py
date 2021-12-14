import numpy as np
import torch
from metric import metric


class ConfusionMatrix(metric.Metric):
    """
    Constructs a confusion matrix for a multi-class classification problems.

    Parameters:
        num_classes: number of classes.
        is_probability: Determines whether the confusion matrix is shown in probability or not.
    """

    def __init__(self, num_classes, is_probability=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.is_probability = is_probability
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, pre, gt):
        """
        Compute the confusion matrix, shape: (K, K), K = class_num.

        The shape of the confusion matrix is K x K, where K is the number of classes.
        The parameters:
            pre (Tensor or nd-array):  an N-tensor/array of integer values between 0 and K-1.
            target (Tensor or nd-array): an N-tensor/array of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(pre):
            pre = pre.cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.cpu().numpy()

        assert pre.shape[0] == gt.shape[0], 'number of targets and predicted outputs do not match'

        if np.ndim(pre) != 1:
            assert pre.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            pre = np.argmax(pre, 1)
        else:
            assert (pre.max() < self.num_classes) and (pre.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(gt) != 1:
            assert gt.shape[1] == self.num_classes, \
                'one hot target does not match size of confusion matrix'
            assert (gt >= 0).all() and (gt <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (gt.sum(1) == 1).all(), 'multi-label setting is not supported'
            gt = np.argmax(gt, 1)
        else:
            assert (gt.max() < self.num_classes) and (gt.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincount 2 arrays together
        x = pre + self.num_classes * gt
        bincount_2d = np.bincount(
            x.astype(np.int64), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confusion matrix of K rows and K columns, K is the class numbers.
            Rows corresponds to ground-truth targets.
            Columns corresponds to predicted targets.
        """
        if self.is_probability:
            conf = self.conf.astype(np.float32)
            # clip a small number to prevent the values from being too small
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf
