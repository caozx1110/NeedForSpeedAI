import utils
from test_ import Test
from train import Train
from metric.iou import IoU
from models.enet import ENet
from args import get_arguments
import transforms as ext_transforms
from data.utils import enet_weighing

import os
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

args = get_arguments()
device = torch.device(args.device)


def load_dataset(dataset):
    """
    Used to load nfs dataset.
    The steps are as following:
        1, Perform data augmentation.
        2, Initialize train, test, val dataset and create corresponding dataloader.
        3, Get class number and color dictionary.
        4, Get the class weight for the criterion.
    """
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    # step 1
    train_trans = transforms.Compose([
        transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3),
        # transforms.RandomEqualize(),
        transforms.ToTensor(),
    ])

    test_trans = transforms.ToTensor()
    label_trans = transforms.Compose([
        ext_transforms.PILToLongTensor(),
    ])
    dual_trans = None
    # dual_trans = transforms.Compose([
    #     transforms.RandomApply(
    #         nn.ModuleList([
    #             transforms.RandomResizedCrop(size=(480, 640)),
    #             transforms.RandomRotation(30)
    #         ]), p=.5
    #     ),
    #     transforms.RandomHorizontalFlip(),
    # ])
    # print(dual_trans)

    # step 2
    train_set = dataset(args.dataset_dir, transform=train_trans,
                        label_transform=label_trans, dual_trans=dual_trans)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers)
    val_set = dataset(args.dataset_dir, mode='val', transform=test_trans,
                      label_transform=label_trans)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)
    test_set = dataset(args.dataset_dir, mode='test', transform=test_trans,
                       label_transform=label_trans)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)

    # step 3
    color_code = train_set.color_encoding
    num_class = len(color_code)

    # Print important information and show for debugging.
    print("Number of classes to predict:", num_class)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))
    images, labels = iter(test_loader).next() if args.mode.lower() == 'test' \
        else iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color dictionary:", color_code)

    if args.imshow_batch:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(color_code),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # Step 4
    print("\nWeighing technique:", args.weighing)
    print("Computing class weights...")
    print("This can take a while depending on the dataset size")
    if args.weighing.lower() == 'enet':
        class_weights = enet_weighing(train_loader, num_class)
    else:
        class_weights = None
    if class_weights is not None:
        # Convert weight to ``FloatTensor`` type and send it to the device.
        class_weights = torch.from_numpy(class_weights).float().to(device)
        if args.ignore_unlabeled:
            ignore_index = list(color_code).index('unlabeled')
            class_weights[ignore_index] = 0
    print("Class weights:", class_weights)

    return (train_loader, val_loader, test_loader), class_weights, color_code


def train(train_loader, val_loader, class_weights, class_encoding):
    """
    Use the class ``train`` to preform training.
    The loss function is CrossEntropy, ENet uses it to do fit the labels.
    The optimizer is adam, which is the same as that the authors of ENet choose.
    The metric is IoU and confusion matrix.
    """
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Initialize ENet
    model = ENet(num_classes).to(device)
    # Check if the network architecture is correct.
    print(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = .0

    # Start Training
    print()
    train = Train(model, train_loader, optimizer, criterion, metric, device)
    val = Test(model, val_loader, criterion, metric, device)
    for epoch in range(start_epoch, args.epochs):
        print("[Epoch: {0}] Training".format(epoch + 1))

        epoch_loss, (iou, miou) = train.run_once(args.print_loss)
        lr_updater.step()

        print("[Epoch: {0}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
              format(epoch + 1, epoch_loss, miou))

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print("[Epoch: {0}] Validation".format(epoch + 1))

            loss, (iou, miou) = val.run_once(args.print_loss)

            print("[Epoch: {0}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".
                  format(epoch + 1, loss, miou))

            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))

            # Save the model if it's the best thus far
            if miou > best_miou:
                print("\nBest model thus far. Saving...\n")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, args)
    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, device)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_once(args.print_loss)

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(images)

    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    predictions = torch.argmax(predictions.data, 1)

    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the dataset directory doesn't exist
    assert os.path.isdir(args.dataset_dir), \
        "The directory \"{0}\" doesn't exist.".format(args.dataset_dir)

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(args.save_dir), \
        "The directory \"{0}\" doesn't exist.".format(args.save_dir)

    # Import the requested dataset
    if args.dataset.lower() == 'camvid':
        from data import CamVid as dataset
    elif args.dataset.lower() == 'cityscapes':
        from data import Cityscapes as dataset
    elif args.dataset.lower() == 'nfs':
        from data.nfs_dataset import nfs_seg_dataset as dataset
    else:
        # Should never happen...but just in case it does
        raise RuntimeError("\"{0}\" is not a supported dataset.".format(args.dataset))

    loaders, w_class, class_encoding = load_dataset(dataset)
    train_loader, val_loader, test_loader = loaders

    if args.mode.lower() in {'train', 'full'}:
        model = train(train_loader, val_loader, w_class, class_encoding)

    if args.mode.lower() in {'test', 'full'}:
        if args.mode.lower() == 'test':
            # Initialize a new ENet model
            num_classes = len(class_encoding)
            model = ENet(num_classes).to(device)

        # Initialize a optimizer just so we can retrieve the model from the checkpoint
        optimizer = optim.Adam(model.parameters())

        # Load the previously saved model state to the ENet model
        model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]

        if args.mode.lower() == 'test':
            print(model)

        test(model, test_loader, w_class, class_encoding)


# train_trans = transforms.Compose([
#     transforms.RandomApply(
#         nn.ModuleList([
#             transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.3),
#             transforms.RandomInvert()
#         ]),
#         p=.5
#     ),
#     transforms.ToTensor(),
# ])
#
# test_trans = transforms.Compose([
#     transforms.ToTensor(),
# ])
# label_trans = transforms.Compose([
#     ext_transforms.PILToLongTensor(),
# ])
# dual_trans = transforms.Compose([
#     transforms.RandomResizedCrop(size=(480, 640)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
# ])

# train_set = nfs_seg_dataset('./data/nfs', transform=train_trans,
#                             label_transform=label_trans, intensity=0, dual_trans=dual_trans)
# i = 0
# i += 1
# plt.close()
# img, lbl = train_set.__getitem__(i)
# lbl = ext_transforms.LongTensorToRGBPIL(nfs_seg_dataset.color_encoding)(lbl)
# plt.subplot(121)
# plt.imshow(lbl)
# plt.subplot(122)
# plt.imshow(img.transpose(0, 1).transpose(1, 2).numpy())
