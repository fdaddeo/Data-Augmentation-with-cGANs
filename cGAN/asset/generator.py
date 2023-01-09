import torch
import torch.nn as NN

class Generator32(NN.Module):
    """
    This class represents the generator of the cGAN for images with a dimension of 32 pixels.
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

        super(Generator32, self).__init__()

        ## Noise block
        to_add = [
            # (bs, latentspace, 1, 1) -> (bs, latentspace * 2, 4, 4)
            NN.ConvTranspose2d(in_channels=config['latentspace_dim'],
                            out_channels=config['featuremap_dim'] * 2,
                            kernel_size=4,
                            stride=1,
                            padding=0,
                            bias=False
                            )
        ]
        
        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 2),
                NN.ReLU(True)
            ]
        elif config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.ReLU(True)
            ]

        self.noise_block = NN.Sequential(*to_add)
        to_add.clear()
        
        ## Label block
        to_add = [
            # (bs, 10, 1, 1) -> (bs, featuremap * 2, 4, 4)
            NN.ConvTranspose2d(in_channels=num_label,
                               out_channels=config['featuremap_dim'] * 2,
                               kernel_size=4, 
                               stride=1,
                               padding=0,
                               bias=False
                              )
        ]

        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 2),
                NN.ReLU(True)
            ]
        elif config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.ReLU(True)
            ]

        self.label_block = NN.Sequential(*to_add)
        to_add.clear()
        
        ## Main block
        to_add = [
            # (bs, featuremap * 4, 4, 4) -> (bs, featuremap * 2, 8, 8)
            NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 4,
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
                NN.ReLU(True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap, 16, 16)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 2,
                                   out_channels=config['featuremap_dim'],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1, 
                                   bias=False
                                  ),
                NN.BatchNorm2d(config['featuremap_dim']),
                NN.ReLU(True),
                # (bs, featuremap, 16, 16) -> (bs, 3, 32, 32)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'],
                                   out_channels=image_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False
                                  ),
                # final shape: (bs, 3, 32, 32)
                NN.Tanh()
            ]
        elif config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.ReLU(True),
                # (bs, featuremap * 2, 8, 8) -> (bs, featuremap, 16, 16)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 2,
                                   out_channels=config['featuremap_dim'],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1, 
                                   bias=False
                                  ),
                NN.InstanceNorm2d(config['featuremap_dim'], affine=True),
                NN.ReLU(True),
                # (bs, featuremap, 16, 16) -> (bs, 3, 32, 32)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'],
                                   out_channels=image_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False
                                  ),
                # final shape: (bs, 3, 32, 32)
                NN.Tanh()     
            ]

        self.main = NN.Sequential(*to_add)
        to_add.clear()

    def forward(self, noise, labels):
        # First lets pass the noise and the labels through the corresponding layers ...
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        
        # ... then concatenate them over the channels and fed the output to the rest of the generator.
        x = torch.cat([z_out, l_out], dim = 1)
        
        return self.main(x)


class Generator64(NN.Module):
    """
    This class represents the generator of the cGAN for images with a dimension of 64 pixels.
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

        super(Generator64, self).__init__()

        ## Noise block
        to_add = [
            # (bs, latentspace, 1, 1) -> (bs, latentspace * 4, 4, 4)
            NN.ConvTranspose2d(in_channels=config['latentspace_dim'],
                            out_channels=config['featuremap_dim'] * 4,
                            kernel_size=4,
                            stride=1,
                            padding=0,
                            bias=False
                            )
        ]
        
        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 4),
                NN.ReLU(True)
            ]
        elif config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 4, affine=True),
                NN.ReLU(True)
            ]

        self.noise_block = NN.Sequential(*to_add)
        to_add.clear()
        
        ## Label block
        to_add = [
            # (bs, 10, 1, 1) -> (bs, featuremap * 4, 4, 4)
            NN.ConvTranspose2d(in_channels=num_label,
                               out_channels=config['featuremap_dim'] * 4,
                               kernel_size=4, 
                               stride=1,
                               padding=0,
                               bias=False
                              )
        ]

        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 4),
                NN.ReLU(True)
            ]
        elif config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 4, affine=True),
                NN.ReLU(True)
            ]

        self.label_block = NN.Sequential(*to_add)
        to_add.clear()
        
        ## Main block
        to_add = [
            # (bs, featuremap * 8, 4, 4) -> (bs, featuremap * 4, 8, 8)
            NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 8,
                               out_channels=config['featuremap_dim'] * 4, 
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                              )
        ]

        if config['use_batch_norm']:
            to_add += [
                NN.BatchNorm2d(config['featuremap_dim'] * 4),
                NN.ReLU(True),
                # (bs, featuremap * 4, 8, 8) -> (bs, featuremap * 2, 16, 16)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 4, 
                                   out_channels=config['featuremap_dim'] * 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1, 
                                   bias=False
                                  ),
                NN.BatchNorm2d(config['featuremap_dim'] * 2),
                NN.ReLU(True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap, 32, 32)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 2, 
                                   out_channels=config['featuremap_dim'],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False
                                  ),
                NN.BatchNorm2d(config['featuremap_dim']),
                NN.ReLU(True),
                # (bs, featuremap, 32, 32) -> (bs, featuremap, 64, 64)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'], 
                                   out_channels=image_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False
                                  ),
                # final shape: (bs, 3, 64, 64)
                NN.Tanh()
            ]
        elif config['use_instance_norm']:
            to_add += [
                NN.InstanceNorm2d(config['featuremap_dim'] * 4, affine=True),
                NN.ReLU(True),
                # (bs, featuremap * 4, 8, 8) -> (bs, featuremap * 2, 16, 16)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 4, 
                                   out_channels=config['featuremap_dim'] * 2,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1, 
                                   bias=False
                                  ),
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True),
                NN.ReLU(True),
                # (bs, featuremap * 2, 16, 16) -> (bs, featuremap, 32, 32)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 2, 
                                   out_channels=config['featuremap_dim'],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False
                                  ),
                NN.InstanceNorm2d(config['featuremap_dim'], affine=True),
                NN.ReLU(True),
                # (bs, featuremap, 32, 32) -> (bs, featuremap, 64, 64)
                NN.ConvTranspose2d(in_channels=config['featuremap_dim'], 
                                   out_channels=image_channels,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False
                                  ),
                # final shape: (bs, 3, 64, 64)
                NN.Tanh()     
            ]

        self.main = NN.Sequential(*to_add)
        to_add.clear()

    def forward(self, noise, labels):
        # First lets pass the noise and the labels through the corresponding layers ...
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        
        # ... then concatenate them over the channels and fed the output to the rest of the generator.
        x = torch.cat([z_out, l_out], dim = 1)
        
        return self.main(x)