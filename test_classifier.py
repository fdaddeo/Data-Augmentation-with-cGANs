import os
import argparse
import torch
import yaml

from torch.utils.data.dataloader import DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

from classifier.tester import Tester

def argparser():
    args = argparse.ArgumentParser()

    args.add_argument('--model_name', type=str, help='File name of the model to be used during testing', required=True)
    args.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'], help='Specify the device on which executes the training.', required=False)
    args.add_argument('--config', type=str, default='./config/classifier/alexnet_cGAN_epoch100.yaml', help='Path to the configuration file.', required=False)

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

    # Resize is fine both for alexnet and resnet
    transformList = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

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

    tester = Tester(test_loader=cifar10_testloader,
                    device=device,
                    args=args,
                    config=config)

    tester.test()

if __name__ == "__main__":
    # To suppress tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = argparser()
    main(args)