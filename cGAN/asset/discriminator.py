import torch
import torch.nn as NN

class Discriminator(NN.Module):
    """
    This class represents the discriminator of the cGAN.
    """

    def __init__(self, args: dict):
        """
        Parameters
        ----------
        args : dict
            The argument parsed on program launch.
        """

        super(Discriminator, self).__init__()

        self.img_block = NN.Sequential(
            NN.Conv2d(in_channels=args.num_image_channels, 
                      out_channels=args.dis_featuremap_dim//2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     ), 
            NN.LeakyReLU(0.2, inplace=True),
        )

        self.label_block = NN.Sequential(
            NN.Conv2d(in_channels=args.num_label, 
                      out_channels=args.dis_featuremap_dim//2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     ),
            NN.LeakyReLU(0.2, inplace=True),
        )

        self.main = NN.Sequential(
            NN.Conv2d(in_channels=args.dis_featuremap_dim, 
                      out_channels=args.dis_featuremap_dim * 2, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     ),
            NN.BatchNorm2d(args.dis_featuremap_dim * 2),
            NN.LeakyReLU(0.2, inplace=True),
            NN.Conv2d(in_channels=args.dis_featuremap_dim * 2, 
                      out_channels=args.dis_featuremap_dim * 4, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False
                     ),
            NN.BatchNorm2d(args.dis_featuremap_dim * 4),
            NN.LeakyReLU(0.2, inplace=True),
            NN.Conv2d(in_channels=args.dis_featuremap_dim * 4, 
                      out_channels=1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False
                     ),
            NN.Sigmoid()
        )

    def forward(self, img, label):
        # First lets pass the images and the labels through the corresponding layers ...
        img_out = self.img_block(img)
        lab_out = self.label_block(label)

        # ... then concatenate them over the channels and fed the output to the rest of the discriminator.
        x = torch.cat([img_out, lab_out], dim = 1)

        return self.main(x)