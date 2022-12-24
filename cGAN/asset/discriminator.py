import torch
import torch.nn as NN

class Discriminator(NN.Module):
    """
    This class represents the discriminator of the cGAN.
    """

    def __init__(self, config: dict, num_label: int, image_channels: int):
        """
        Parameters
        ----------
        config : dict
            The options for the generator contained in the configuration file.

        num_label : int
            The number of labels.

        image_channels : int
            The number of channels of the images.
        """

        # TODO: implement batch norm as generator

        super(Discriminator, self).__init__()

        self.img_block = NN.Sequential(
            NN.Conv2d(in_channels=image_channels, 
                      out_channels=config['featuremap_dim'] // 2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     ), 
            NN.LeakyReLU(0.2, inplace=True),
        )

        self.label_block = NN.Sequential(
            NN.Conv2d(in_channels=num_label, 
                      out_channels=config['featuremap_dim'] // 2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     ),
            NN.LeakyReLU(0.2, inplace=True),
        )

        self.main = NN.Sequential(
            NN.Conv2d(in_channels=config['featuremap_dim'], 
                      out_channels=config['featuremap_dim'] * 2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     )
        )

        if config['use_batch_norm']:
            self.main.append(
                NN.BatchNorm2d(config['featuremap_dim'] * 2)
            )
        elif config['use_instance_norm']:
            self.main.append(
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True)
            )

        self.main.append(
            NN.LeakyReLU(0.2, inplace=True)
        )
        self.main.append(
            NN.Conv2d(in_channels=config['featuremap_dim'] * 2, 
                      out_channels=config['featuremap_dim'] * 4, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     )
        )

        if config['use_batch_norm']:
            self.main.append(
                NN.BatchNorm2d(config['featuremap_dim'] * 4)
            )
        elif config['use_instance_norm']:
            self.main.append(
                NN.InstanceNorm2d(config['featuremap_dim'] * 4, affine=True)
            )

        self.main.append(
            NN.LeakyReLU(0.2, inplace=True)
        )
        self.main.append(
            NN.Conv2d(in_channels=config['featuremap_dim'] * 4, 
                      out_channels=1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False
                     )
        )

        self.main.append(
            NN.Sigmoid()
        )

    def forward(self, img, label):
        # First lets pass the images and the labels through the corresponding layers ...
        img_out = self.img_block(img)
        lab_out = self.label_block(label)

        # ... then concatenate them over the channels and fed the output to the rest of the discriminator.
        x = torch.cat([img_out, lab_out], dim = 1)

        return self.main(x)