import torch
import numpy as np
from metric import metric
from metric.confusionmatrix import ConfusionMatrix


class IoU(metric.Metric):
    """
    Computes the intersection over union (IoU) per class and corresponding mean (mIoU).

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Parameters:
        num_classes: number of classes in the classification problem
        is_probability: Determines whether the confusion matrix is shown in probability or not.
        ignore_index: Index of the classes to ignore when computing the IoU.
    """

    def __init__(self, num_classes, is_probability=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, is_probability)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, pre, gt):
        """
        Add the predicted and target pair to the IoU metric.
        The parameters:
            pre: tensor of integer values between 0 and K-1, shape: (N, H, W).
            gt: tensor of integer values between 0 and K-1. shape: (N, H, W).
        """
        # Dimensions check
        assert pre.size(0) == gt.size(0), 'Number of targets and predicted outputs do not match'
        assert pre.dim() == 3, 'Predictions must be of dimension (N, H, W)'
        assert gt.dim() == 3, 'Targets must be of dimension (N, H, W)'

        self.conf_metric.add(pre.view(-1), gt.view(-1))

    def value(self):
        """
        Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU).
            The first output is the per class IoU, shape: (K, )
            The second output is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return iou, np.nanmean(iou)
