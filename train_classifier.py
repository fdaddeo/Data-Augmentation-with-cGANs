import os
import argparse
import torch
import yaml

import numpy as np

from torch.utils.data import ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10, ImageFolder

from classifier.trainer import FineTune

def argparser():
    args = argparse.ArgumentParser()

    args.add_argument('--run_name', type=str, help='The name of the run will be used to store the results.', required=True)
    args.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'],help='Specify the device on which executes the training.', required=False)
    args.add_argument('--config', type=str, default='./config/classifier/basic_classifier_epoch100.yaml', help='Path to the configuration file.', required=False)
    args.add_argument('--augment_data', action='store_true', help='Perform data augmentation of CIFAR10.')

    return args.parse_args()

def get_config(config: str):
    """
    Load the configuration file.

    Parameters
    ----------
    config : str
        Path of the configuration file.

    Returns
    -------
    The yaml configuration file parsed.
    """

    with open(config, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main(args):
    config = get_config(args.config)
    writer = SummaryWriter(os.path.join(config['tensorboard_path'], args.run_name))

    os.makedirs(config['model_path'], exist_ok=True)

    # Required by vgg
    transformList = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    if args.augment_data:
        cifar10_trainset = CIFAR10(root=config['dataset_path'],
                                train=True,
                                transform=transformList,
                                download=True)

        augment_trainset = ImageFolder(root=config['generated_dataset_path'],
                                       transform=transformList)

        cifar10_len = len(cifar10_trainset)
        augment_len = len(augment_trainset)
        split = int(0.7 * cifar10_len)

        cifar10_train, _ = random_split(cifar10_trainset, [split, cifar10_len - split])
        augment_train, _ = random_split(augment_trainset, [augment_len - split, split])

        trainset = ConcatDataset([cifar10_train, augment_train])
    else:
        trainset = ImageFolder(root=config['generated_dataset_path'],
                               transform=transformList)

    trainloader = DataLoader(dataset=trainset,
                             batch_size=config['batch_size'],
                             shuffle=True,
                             num_workers=config['num_workers'])

    cifar10_testset = CIFAR10(root=config['dataset_path'],
                              train=False,
                              transform=transformList,
                              download=True)

    cifar10_testloader = DataLoader(dataset=cifar10_testset,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    num_workers=config['num_workers'])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Code will be executed on {device}")

    trainer = FineTune(writer=writer,
                       train_loader=trainloader,
                       test_loader=cifar10_testloader,
                       device=device,
                       args=args,
                       config=config)

    trainer.train()

if __name__ == "__main__":
    # To suppress tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = argparser()
    main(args)