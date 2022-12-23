import os
import torch
import torch.nn as NN
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from .asset.generator import Generator
from .asset.discriminator import Discriminator

class Trainer(object):
    """
    This class implements the training for the cGAN.
    """

    def __init__(self, writer: SummaryWriter, train_loader: DataLoader, device: torch.device, args: dict):
        """
        Parameters
        ----------
        writer : SummaryWriter
            The SummaryWriter for storing infos about training in a tensorboard format.

        train_loader : DataLoader
            The DataLoader for the train images.

        args : dict
            The arguments parsed on program launch.
        """

        self.writer = writer
        self.train_loader = train_loader
        self.device = device
        self.args = args

        self.generator_name = f"cgan_gen_{self.args.run_name}"
        self.discriminator_name = f"cgan_dis_{self.args.run_name}"
        self.generator = Generator(self.args).to(self.device)
        self.discriminator = Discriminator(self.args).to(self.device)

        # Define Loss function
        if self.args.loss == 'BCE':
            self.criterion = NN.BCELoss()
        elif self.args.loss == 'WAS':
            self.criterion = "WAS"

        # Define Optimizer
        if self.args.optimizer == 'Adam':
            self.optimizerG = optim.Adam(self.generator.parameters(), lr=args.learning_rate, betas=(args.beta, 0.999))
            self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=(args.beta, 0.999))
        elif self.args.optimizer == 'SGD':
            self.optimizerG = optim.SGD(self.generator.parameters(), lr=args.learning_rate, momentum=0.9)
            self.optimizerD = optim.SGD(self.discriminator.parameters(), lr=args.learning_rate, momentum=0.9)

    def save_model(self):
        """
        Function that saves the model.
        """

        pathG = os.path.join(self.args.gen_checkpoint_path, self.generator_name)
        pathD = os.path.join(self.args.dis_checkpoint_path, self.discriminator_name)
        torch.save(self.generator.state_dict(), pathG)
        print("Generator Model saved!")
        torch.save(self.discriminator.state_dict(), pathD)
        print("Discriminator Model saved!")

    def weights_init(self, m):
        """
        It performs a custom initialization of the weights. This function has to be call either on Generator and Discriminator because
        on the original paper of cGAN the authors specified that all model weights shall be randomly initialized from a Normal distribution 
        with mean=0 and stdev=0.02.
        """

        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            NN.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            NN.init.normal_(m.weight.data, 1.0, 0.02)
            NN.init.constant_(m.bias.data, 0)

    def gradient_penalty(self, y, x):
        """
        Function used when it has been selected the Wassertein Loss. It computes the gradient penalty as (L2_norm(dy/dx) - 1)**2.
        """

        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True
                                  )[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))

        return torch.mean((dydx_l2norm - 1)**2)

    def train(self):
        # TODO
        # TODO implement early stopping and ReduceLROnPlateau
        return