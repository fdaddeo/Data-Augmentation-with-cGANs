import os
import torch

import torch.nn as NN
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

class FineTune(object):
    """
    This class implements the training for the pretrained classifier.
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

        self.model = models.inception_v3(weights='DEFAULT')

        for idx, module in enumerate(self.model.named_children()):
            if module[0] != 'fc':
                self.set_parameter_requires_grad(module[1], False)

        num_feature = self.model.fc.in_features
        self.model.fc = NN.Linear(num_feature, self.config['num_label'])
        
        self.model.to(device)

    def save_models(self, epoch: int):
        """
        Function that saves the model.

        Parameters
        ----------
        epoch : int
            The epoch in which the model was saved.
        """

        path = os.path.join(self.config['model_path'], f"{self.model_name}_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), path)
        print("Model saved!")

    def set_parameter_requires_grad(self, model, req_grad=False):
        """"
        Function that freezes the model layers.

        Parameters
        ----------
        req_grad : bool
            Set False if you want to freeze the layer, True otherwise.
        """

        for param in model.parameters():
            param.requires_grad = req_grad

    def train(self):
        return
