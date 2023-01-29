import os
import argparse
import torch
import yaml

from torch.utils.data.dataloader import DataLoader

from cGAN.tester import Tester

def argparser():
    args = argparse.ArgumentParser()

    args.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'],help='Specify the device on which executes the training.', required=False)
    args.add_argument('--config', type=str, default='./config/cgan/basic_cGAN.yaml', help='Path to the configuration file.', required=False)

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
        file = yaml.load(f, Loader=yaml.FullLoader)
        
        if (file['gen']['use_batch_norm'] and file['gen']['use_instance_norm']):
            raise Exception("ERROR: in Generator and Discriminator 'use_batch_norm' and 'use_instance_norm' cannot be both true. Please fix the configuration file.")

        if len(file['classes']) != file['num_label']:
            raise Exception("ERROR: 'num_label' and 'classes' must have the same dimension. Please fix the configuration file.")

        return file


def main(args):
    config = get_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Code will be executed on {device}")

    tester = Tester(device=device,
                    config=config)

    tester.generate()

if __name__ == "__main__":
    # To suppress tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = argparser()
    main(args)