import torch
import torch.nn as NN

class Discriminator32(NN.Module):
    """
    This class represents the discriminator of the cGAN for images with a dimension of 32 pixels.
    """

    def __init__(self, config: dict, num_label: int, image_channels: int, loss: str):
        """
        Parameters
        ----------
        config : dict
            The options for the generator contained in the configuration file.

        num_label : int
            The number of labels.

        image_channels : int
            The number of channels of the images.

        loss : str
            The loss specified in the configuration file.
        """

        super(Discriminator32, self).__init__()

        ## Image block
        self.img_block = NN.Sequential(
            # (bs, 3, 32, 32) -> (bs, featuremap / 2, 16, 16)
            NN.Conv2d(in_channels=image_channels,
                      out_channels=config['featuremap_dim'] // 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     ), 
            NN.LeakyReLU(0.2, inplace=True),
        )

        ## Label block
        self.label_block = NN.Sequential(
            # (bs, 10, 32, 32) -> (bs, featuremap / 2, 16, 16)
            NN.Conv2d(in_channels=num_label,
                      out_channels=config['featuremap_dim'] // 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     ),
            NN.LeakyReLU(0.2, inplace=True),
        )

        ## Main block
        to_add = [
            # (bs, featuremap, 16, 16) -> (bs, featuremap * 2, 8, 8)
            NN.Conv2d(in_channels=config['featuremap_dim'],
                      out_channels=config['featuremap_dim'] * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     )
        ]

        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 2),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap * 4, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.BatchNorm2d(config['featuremap_dim'] * 4),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]
        
        if config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap * 4, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 4, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]
        
        if (not config['use_batch_norm']) and (not config['use_instance_norm']):
            to_add += [
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap * 4, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]

        # BCEWithLogits implements itself a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer.
        if loss != 'Wassertein' and loss != 'BCEWithLogits':
            to_add += [
                NN.Sigmoid()
            ]

        self.main = NN.Sequential(*to_add)
        to_add.clear()

    def forward(self, img, label):
        # First lets pass the images and the labels through the corresponding layers ...
        img_out = self.img_block(img)
        lab_out = self.label_block(label)

        # ... then concatenate them over the channels and fed the output to the rest of the discriminator.
        x = torch.cat([img_out, lab_out], dim = 1)

        return self.main(x)


class Discriminator64(NN.Module):
    """
    This class represents the discriminator of the cGAN for images with a dimension of 64 pixels.
    """

    def __init__(self, config: dict, num_label: int, image_channels: int, loss: str):
        """
        Parameters
        ----------
        config : dict
            The options for the generator contained in the configuration file.

        num_label : int
            The number of labels.

        image_channels : int
            The number of channels of the images.

        loss : str
            The loss specified in the configuration file.
        """

        super(Discriminator64, self).__init__()

        ## Image block
        self.img_block = NN.Sequential(
            # (bs, 3, 64, 64) -> (bs, featuremap / 2, 32, 32)
            NN.Conv2d(in_channels=image_channels,
                      out_channels=config['featuremap_dim'] // 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     ), 
            NN.LeakyReLU(0.2, inplace=True),
        )

        ## Label block
        self.label_block = NN.Sequential(
            # (bs, 10, 64, 64) -> (bs, featuremap / 2, 32, 32)
            NN.Conv2d(in_channels=num_label,
                      out_channels=config['featuremap_dim'] // 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     ),
            NN.LeakyReLU(0.2, inplace=True),
        )

        ## Main block
        to_add = [
            # (bs, featuremap, 32, 32) -> (bs, featuremap * 2, 16, 16)
            NN.Conv2d(in_channels=config['featuremap_dim'],
                      out_channels=config['featuremap_dim'] * 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     )
        ]

        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 2),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap * 4, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.BatchNorm2d(config['featuremap_dim'] * 4),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 8, 8) -> (bs, featuremap * 8, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=config['featuremap_dim'] * 8,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.BatchNorm2d(config['featuremap_dim'] * 8),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 8, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 8,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]
        
        if config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap * 4, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 4, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 8, 8) -> (bs, featuremap * 8, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=config['featuremap_dim'] * 8,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 8, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 8, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 8,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]
        
        if (not config['use_batch_norm']) and (not config['use_instance_norm']):
            to_add += [
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap * 4, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 8, 8) -> (bs, featuremap * 8, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=config['featuremap_dim'] * 8,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 8, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 8,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]

        if loss != 'Wassertein' and loss != 'BCEWithLogits':
            to_add += [
                NN.Sigmoid()
            ]

        self.main = NN.Sequential(*to_add)
        to_add.clear()

    def forward(self, img, label):
        # First lets pass the images and the labels through the corresponding layers ...
        img_out = self.img_block(img)
        lab_out = self.label_block(label)

        # ... then concatenate them over the channels and fed the output to the rest of the discriminator.
        x = torch.cat([img_out, lab_out], dim = 1)

        return self.main(x)

class DiscriminatorCustom(NN.Module):
    """
    This class represents the discriminator of the cGAN for images with a dimension of 32 pixels. In particular, for each layer 
    has been introduced an intermediate layer that preserves the dimension.
    """

    def __init__(self, config: dict, num_label: int, image_channels: int, loss: str):
        """
        Parameters
        ----------
        config : dict
            The options for the generator contained in the configuration file.

        num_label : int
            The number of labels.

        image_channels : int
            The number of channels of the images.

        loss : str
            The loss specified in the configuration file.
        """

        super(DiscriminatorCustom, self).__init__()

        ## Image block
        self.img_block = NN.Sequential(
            # (bs, 3, 32, 32) -> (bs, featuremap / 2, 16, 16)
            NN.Conv2d(in_channels=image_channels,
                      out_channels=config['featuremap_dim'] // 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     ), 
            NN.LeakyReLU(0.2, inplace=True),
        )

        ## Label block
        self.label_block = NN.Sequential(
            # (bs, 10, 32, 32) -> (bs, featuremap / 2, 16, 16)
            NN.Conv2d(in_channels=num_label,
                      out_channels=config['featuremap_dim'] // 2,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False
                     ),
            NN.LeakyReLU(0.2, inplace=True),
        )

        ## Main block
        to_add = [
            # (bs, featuremap, 16, 16) -> (bs, featuremap * 2, 16, 16)
            NN.Conv2d(in_channels=config['featuremap_dim'],
                      out_channels=config['featuremap_dim'] * 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False
                     )
        ]

        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 2),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap * 2, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.BatchNorm2d(config['featuremap_dim'] * 2),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap * 3, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 3,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False
                         ),
                NN.BatchNorm2d(config['featuremap_dim'] * 3),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 3, 8, 8) -> (bs, featuremap * 3, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 3,
                          out_channels=config['featuremap_dim'] * 3,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.BatchNorm2d(config['featuremap_dim'] * 3),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 3, 4, 4) -> (bs, featuremap * 4, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 3,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False
                         ),
                NN.BatchNorm2d(config['featuremap_dim'] * 4),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]
        
        if config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap * 2, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap * 3, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 3,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False
                         ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 3, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 3, 8, 8) -> (bs, featuremap * 3, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 3,
                          out_channels=config['featuremap_dim'] * 3,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 3, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 3, 4, 4) -> (bs, featuremap * 4, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 3,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False
                         ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 4, affine=True),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]
        
        if (not config['use_batch_norm']) and (not config['use_instance_norm']):
            to_add += [
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap * 2, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap * 3, 8, 8)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 2,
                          out_channels=config['featuremap_dim'] * 3,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False
                         ),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 3, 8, 8) -> (bs, featuremap * 3, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 3,
                          out_channels=config['featuremap_dim'] * 3,
                          kernel_size=4,
                          stride=2,
                          padding=1,
                          bias=False
                         ),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 3, 4, 4) -> (bs, featuremap * 4, 4, 4)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 3,
                          out_channels=config['featuremap_dim'] * 4,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False
                         ),
                NN.LeakyReLU(0.2, inplace=True),
                # (bs, featuremap * 4, 4, 4) -> (bs, 1, 1, 1)
                NN.Conv2d(in_channels=config['featuremap_dim'] * 4,
                          out_channels=1,
                          kernel_size=4,
                          stride=1,
                          padding=0,
                          bias=False
                         )
                # final shape: (bs, 1, 1, 1)
            ]

        # BCEWithLogits implements itself a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer.
        if loss != 'Wassertein' and loss != 'BCEWithLogits':
            to_add += [
                NN.Sigmoid()
            ]

        self.main = NN.Sequential(*to_add)
        to_add.clear()

    def forward(self, img, label):
        # First lets pass the images and the labels through the corresponding layers ...
        img_out = self.img_block(img)
        lab_out = self.label_block(label)

        # ... then concatenate them over the channels and fed the output to the rest of the discriminator.
        x = torch.cat([img_out, lab_out], dim = 1)

        return self.main(x)