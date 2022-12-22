import torch
import torch.nn as NN

class Generator(NN.Module):
    """
    This class represents the generator of the cGAN.
    """

    def __init__(self, args: dict):
        """
        Parameters
        ----------
        args : dict
            The argument parsed on program launch.
        """

        super(Generator, self).__init__()

        self.noise_block = NN.Sequential(
            NN.ConvTranspose2d(in_channels=args.latentspace_dim,
                               out_channels=args.gen_featuremap_dim * 2,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False
                              ),
            NN.BatchNorm2d(args.gen_featuremap_dim * 2),
            NN.ReLU(True)
        )
        
        self.label_block = NN.Sequential(
            NN.ConvTranspose2d(in_channels=args.num_label,
                               out_channels=args.gen_featuremap_dim * 2,
                               kernel_size=4, 
                               stride=1,
                               padding=0,
                               bias=False
                              ),
            NN.BatchNorm2d(args.gen_featuremap_dim * 2),
            NN.ReLU(True)
        )
        
        self.main = NN.Sequential(
            NN.ConvTranspose2d(in_channels=args.gen_featuremap_dim * 4,
                               out_channels=args.gen_featuremap_dim * 2, 
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                              ),
            NN.BatchNorm2d(args.gen_featuremap_dim * 2),
            NN.ReLU(True),

            NN.ConvTranspose2d(in_channels=args.gen_featuremap_dim * 2, 
                               out_channels=args.gen_featuremap_dim,
                               kernel_size=4,
                               stride=2,
                               padding=1, 
                               bias=False
                              ),
            NN.BatchNorm2d(args.gen_featuremap_dim),
            NN.ReLU(True),
            
            NN.ConvTranspose2d(in_channels=args.gen_featuremap_dim,
                               out_channels=args.num_image_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                              ),
            NN.Tanh()
        )

    def forward(self, noise, labels):
        # First lets pass the noise and the labels through the corresponding layers ...
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        
        # ... then concatenate them over the channels and fed the output to the rest of the generator.
        x = torch.cat([z_out, l_out], dim = 1)
        
        return self.main(x)