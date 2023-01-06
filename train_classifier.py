import os
import argparse
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

from classifier.trainer import FineTune

def argparser():
    args = argparse.ArgumentParser()

    args.add_argument('--run_name', type=str, help='The name of the run will be used to store the results.', required=True)
    args.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'],help='Specify the device on which executes the training.', required=False)
    args.add_argument('--config', type=str, default='./config/classifier/sample.yaml', help='Path to the configuration file.', required=False)

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

    # Required by inception_v3
    transformList = transforms.Compose([# transforms.Resize(config['image_size']),
                                        transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616))])

    cifar10_trainset = CIFAR10(root=config['dataset_path'],
                              train=True,
                              transform=transformList,
                              download=True)

    cifar10_testset = CIFAR10(root=config['dataset_path'],
                              train=False,
                              transform=transformList,
                              download=True)

    cifar10_trainloader = DataLoader(dataset=cifar10_trainset,
                                    batch_size=config['batch_size'],
                                    shuffle=True,
                                    num_workers=config['num_workers'])

    cifar10_testloader = DataLoader(dataset=cifar10_testset,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    num_workers=config['num_workers'])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Code will be executed on {device}")

    trainer = FineTune(writer=writer,
                       train_loader=cifar10_trainloader,
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