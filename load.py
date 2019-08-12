import argparse
from torch.utils.data import DataLoader
from data.data_loader import DataManager
from data.utils import visual_img_label


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("ds_name", help="Name of the dataset.")

    # DataLoader
    parser.add_argument("--crop_method", type=str, choices=['sliding', 'random'],
                        help="Patch crop strategy.")
    parser.add_argument("--crop_size", type=int,
                        help="Size of cropped patches.")
    parser.add_argument("--resize_size", type=int,
                        help="Size of resizing patches.")
    parser.add_argument("--pad_size_x", type=int,
                        help="Padding patches along the x axis.")
    parser.add_argument("--pad_size_y", type=int,
                        help="Padding patches along the y axis.")
    parser.add_argument("--augment", action='store_true',
                        help="Whether to augment data")

    # Model
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Minibatch size.")
    args = parser.parse_args()
    return args


def main():
    # Parse arguments.
    args = parse_args()

    # Prepare data loader.
    dataset_args = dict(
        ds_name=args.ds_name,
        crop_method=args.crop_method,
        crop_size=(args.crop_size, args.crop_size),
        resize_size=(args.resize_size, args.resize_size),
        pad_size=(args.pad_size_x, args.pad_size_y),
        augment=['hflip', 'vflip', 'rot90'] if args.augment else None
    )
    train_dataset = DataManager(div='train', **dataset_args)
    valid_dataset = DataManager(div='validation', **dataset_args)

    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=True, num_workers=0)

    print('\nTraining Start!')
    for epoch in range(100):
        num_train_data_used_per_epoch = 0
        while num_train_data_used_per_epoch < train_dataset.num_patch_per_epoch:
            for step, (train_x, train_y) in enumerate(train_dataloader):
                train_x, train_y = train_x.numpy(), train_y.numpy()
                num_batch = train_x.shape[0]
                num_train_data_used_per_epoch += num_batch

                # Train Process...
                visual_img_label(train_x, train_y, dtype='NHWC')
        print('[Epoch {:03d}]'. format(epoch + 1))

        num_valid_data_used_per_epoch = 0
        while num_valid_data_used_per_epoch < valid_dataset.num_patch_per_epoch:
            for _, (valid_x, valid_y) in enumerate(valid_dataloader):
                valid_x, valid_y = valid_x.numpy(), valid_y.numpy()
                num_batch = valid_x.shape[0]
                num_valid_data_used_per_epoch += num_batch

                # Validation Process...
        print('[Epoch {:03d}] - evaluation'. format(epoch + 1))


    dataset_args = dict(
        ds_name=args.ds_name,
        crop_method=None,
        crop_size=None,
        resize_size=(args.resize_size, args.resize_size),
        pad_size=(args.pad_size_x, args.pad_size_y),
        augment=['hflip', 'vflip', 'rot90'] if args.augment else None
    )
    test_dataset = DataManager(div='test', **dataset_args)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)
    for step, (test_x, test_y) in enumerate(test_dataloader):
        test_x, test_y = test_x.numpy(), test_y.numpy()

        # Test Process
    print('\nTest Done\n')


if __name__ == '__main__':
    main()