import torch
import torch.nn as NN

class Generator(NN.Module):
    """
    This class represents the generator of the cGAN.
    """

    def __init__(self, config: dict, args: dict, num_label: int, image_channels: int):
        """
        Parameters
        ----------
        config : dict
            The options for the generator contained in the configuration file.

        args : dict
            The arguments passed as program launch.

        num_label : int
            The number of labels.

        image_channels : int
            The number of channels of the images.
        """

        super(Generator, self).__init__()

        # Noise block
        self.noise_block = NN.Sequential(
            NN.ConvTranspose2d(in_channels=config['latentspace_dim'],
                               out_channels=config['featuremap_dim'] * 2,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False
                              )
        )
        
        if config['use_batch_norm']:
            self.noise_block.append(
                NN.BatchNorm2d(config['featuremap_dim'] * 2)
            )
        elif config['use_instance_norm']:
            self.noise_block.append(
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True)
            )

        self.noise_block.append(NN.ReLU(True))
        
        # Label block
        self.label_block = NN.Sequential(
            NN.ConvTranspose2d(in_channels=num_label,
                               out_channels=config['featuremap_dim'] * 2,
                               kernel_size=4, 
                               stride=1,
                               padding=0,
                               bias=False
                              )
        )

        if config['use_batch_norm']:
            self.label_block.append(
                NN.BatchNorm2d(config['featuremap_dim'] * 2)
            )
        elif config['use_instance_norm']:
            self.label_block.append(
                NN.InstanceNorm2d(config['featuremap_dim'] * 2, affine=True)
            )
            
        self.label_block.append(
            NN.ReLU(True)
        )
        
        # Main block
        self.main = NN.Sequential(
            NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 4,
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
            NN.ReLU(True)
        )
        self.main.append(
            NN.ConvTranspose2d(in_channels=config['featuremap_dim'] * 2, 
                               out_channels=config['featuremap_dim'],
                               kernel_size=4,
                               stride=2,
                               padding=1, 
                               bias=False
                              )
        )

        if config['use_batch_norm']:
            self.main.append(
                NN.BatchNorm2d(config['featuremap_dim'])
            )
        elif config['use_instance_norm']:
            self.main.append(
                NN.InstanceNorm2d(config['featuremap_dim'], affine=True)
            )

        self.main.append(
            NN.ReLU(True)
        )
        self.main.append(            
            NN.ConvTranspose2d(in_channels=config['featuremap_dim'],
                               out_channels=image_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                              )
        )
        self.main.append(
            NN.Tanh()
        )

    def forward(self, noise, labels):
        # First lets pass the noise and the labels through the corresponding layers ...
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        
        # ... then concatenate them over the channels and fed the output to the rest of the generator.
        x = torch.cat([z_out, l_out], dim = 1)
        
        return self.main(x)