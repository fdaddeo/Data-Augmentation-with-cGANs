import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

class Trainer(object):
    """
    This class implements the training for the cGAN.
    """

    def __init__(self, writer: SummaryWriter, train_loader: DataLoader, device: torch.device, args: dict, config: dict):
        """
        Parameters
        ----------
        writer : SummaryWriter
            The SummaryWriter for storing infos about training in a tensorboard format.

        train_loader : DataLoader
            The DataLoader for the train images.

        args : dict
            The arguments passed as program launch.

        config : dict
            The configuration file.
        """

        self.writer = writer
        self.train_loader = train_loader
        self.device = device
        self.args = args
        self.config = config

        self.model_name = f"classificator_{self.args.run_name}"

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='DEFAULT')
