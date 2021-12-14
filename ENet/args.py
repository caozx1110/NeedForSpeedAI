from argparse import ArgumentParser


def get_arguments():
    """
    Define the command-line arguments, and parses them.
    The main categories of arguments are: mode, hyper parameter, dataset parameter and
        some other settings about folders etc.
    Default training method:
        total epoch numbers is 300, the initial learning rate is 5e-4 and will decay by a
        factor of .1 every 100 epoch training.
    """
    parser = ArgumentParser()

    # Choose mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=("train: performs training and validation; test: tests the model "
              "found in \"--save-dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        action='store_true',
        help=("The model found in \"--save-dir/--name/\" and filename "
              "\"--name.h5\" is loaded."))

    # Add hyper parameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=.1,
        help="The learning rate decay factor. Default: 0.1"
    )
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=100,
        help="The number of epochs before adjusting the learning rate. Default: 100"
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4"
    )

    # Add dataset parameters
    parser.add_argument(
        "--dataset",
        choices=['nfs', 'camvid', 'cityscapes'],
        default='nfs',
        help="Dataset to use. Default: nfs"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/nfs",
        help="Path to the root directory of the selected dataset. "
        "Default: data/nfs"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="The image height. Default: 480"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="The image width. Default: 640"
    )
    parser.add_argument(
        "--weighing",
        choices=['enet', 'none'],
        default='ENet',
        help="The class weighing technique to apply to the dataset. "
        "Default: enet"
    )
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="The unlabeled class is not ignored."
    )

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4"
    )
    parser.add_argument(
        "--print-loss",
        action='store_true',
        help="Print loss every iteration"
    )
    parser.add_argument(
        "--imshow-batch",
        action='store_true',
        help="Displays batch images when loading the dataset and making predictions."
    )
    parser.add_argument(
        "--device",
        default='cuda',
        help="Device on which the network will be trained. Default: cuda"
    )
    parser.add_argument(
        "--name",
        type=str,
        default='ENet',
        help="Name given to the model when saving. Default: ENet"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default='save',
        help="The directory where models are saved. Default: save"
    )

    return parser.parse_args()
