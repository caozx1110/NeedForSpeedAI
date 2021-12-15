import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def batch_transform(batch, transform):
    """
    Apply a series of transforms to a batch of samples.
    Parameters:
        batch: a batch os samples
        transform: A function/transform to apply to ``batch``
    """
    trans_slices = [transform(tensor) for tensor in torch.unbind(batch)]
    return torch.stack(trans_slices)


def imshow_batch(images, labels):
    """
    Display two grids of images. The top grid displays ``images`` and the bottom grid ``labels``

    Keyword arguments:
    - images: a 4D mini-batch tensor of shape (B, C, H, W)
    - labels: a 4D mini-batch tensor of shape (B, C, H, W)
    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))
    plt.show()


def save_checkpoint(model, optimizer, epoch, miou, args):
    """
    Save the model in a specified directory with a specified name.save

    Parameters:
        optimizer: The optimizer state to save.
        epoch: The current epoch for the model.
        miou: The best mean IoU obtained by the model.
        args: An instance of ArgumentParser which contains the arguments used to train ``model``.
        The arguments are written to a text file in ``args.save_dir`` named "``args.name``_args.txt".
    """
    name = args.name
    save_dir = args.save_dir

    assert os.path.isdir(
        save_dir), "The directory \"{0}\" doesn't exist.".format(save_dir)

    # Save model
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)

    # Save arguments
    summary_filename = os.path.join(save_dir, name + '_summary.txt')
    with open(summary_filename, 'w') as summary_file:
        sorted_args = sorted(vars(args))
        summary_file.write("ARGUMENTS\n")
        for arg in sorted_args:
            arg_str = "{0}: {1}\n".format(arg, getattr(args, arg))
            summary_file.write(arg_str)

        summary_file.write("\nBEST VALIDATION\n")
        summary_file.write("Epoch: {0}\n". format(epoch))
        summary_file.write("Mean IoU: {0}\n". format(miou))


def load_checkpoint(model, optimizer, folder_dir, filename):
    """
    Save the model in a specified directory with a specified name.save

    Parameter:
        model: The stored model state is copied to this model instance.
        optimizer: The stored optimizer state is copied to this optimizer instance.
        folder_dir: The path to the folder where the saved model state is located.
        filename: The model filename.

    Returns:
    The epoch, mean IoU, ``model``, and ``optimizer`` loaded from thecheckpoint.
    """
    assert os.path.isdir(folder_dir), \
        "The directory \"{0}\" doesn't exist.".format(folder_dir)

    # Create folder to save model and information
    model_path = os.path.join(folder_dir, filename)
    assert os.path.isfile(model_path), \
        "The model file \"{0}\" doesn't exist.".format(filename)

    # Load the stored model parameters to the model instance
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']

    return model, optimizer, epoch, miou