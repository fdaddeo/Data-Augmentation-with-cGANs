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
            NN.ConvTranspose2d(args.latentspace_dim, args.gen_featuremap_dim * 2, 4, 1, 0, bias=False), # bs, 100, 1, 1 -> bs, args.gen_featuremap_dim * 2, 4, 4
            NN.BatchNorm2d(args.gen_featuremap_dim * 2),
            NN.ReLU(True)
         )
        
        self.label_block = NN.Sequential(
            NN.ConvTranspose2d(args.num_label, args.gen_featuremap_dim * 2, 4, 1, 0, bias=False), #bs, 10, 1, 1 -> bs, args.gen_featuremap_dim * 2, 4, 4
            NN.BatchNorm2d(args.gen_featuremap_dim * 2),
            NN.ReLU(True)
         )
        
        self.main = NN.Sequential(
            NN.ConvTranspose2d( args.gen_featuremap_dim * 4, args.gen_featuremap_dim * 2, 4, 2, 1, bias=False),
            NN.BatchNorm2d(args.gen_featuremap_dim * 2),
            NN.ReLU(True),

            NN.ConvTranspose2d( args.gen_featuremap_dim * 2, args.gen_featuremap_dim, 4, 2, 1, bias=False),
            NN.BatchNorm2d(args.gen_featuremap_dim),
            NN.ReLU(True),
            
            NN.ConvTranspose2d(args.gen_featuremap_dim, args.num_image_channels, 4, 2, 1, bias=False),
            NN.Tanh()
        )

    def forward(self, noise, labels):
        # first lets pass the noise and the labels...
        # through the corresponding layers
        z_out = self.noise_block(noise)
        l_out = self.label_block(labels)
        # then concatenate them and fed the output to the rest of the generator
        x = torch.cat([z_out, l_out], dim = 1) # concatenation over chaNNels
        # bs, args.gen_featuremap_dim*4, 4, 4
        return self.main(x)