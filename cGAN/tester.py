import os
import torch

import torchvision.utils as vutils

from .asset.generator import Generator32, Generator64

class Tester(object):
    """
    This class implements testing for the cGAN.
    """

    def __init__(self, device: torch.device, args: dict, config: dict):
        """
        Parameters
        ----------
        device : torch.device
            Device in which execute the code.

        args : dict
            The arguments passed as program launch.

        config : dict
            The configuration file.
        """

        self.device = device
        self.args = args
        self.config = config

        # Initialize Models
        if self.config['image_size'] == 32:
            self.generator = Generator32(config=self.config['gen'],
                                         num_label=self.config['num_label'],
                                         image_channels=self.config['image_channels']
                                        ).to(self.device)
        elif self.config['image_size'] == 64:
            self.generator = Generator64(config=self.config['gen'],
                                         num_label=self.config['num_label'],
                                         image_channels=self.config['image_channels']
                                        ).to(self.device)
        else:
            raise Exception(f"Image size {self.config['image_size']} not implemented. Please fix the configuration file.")

        # Load trained models
        self.generator.load_state_dict(torch.load(self.config['gen_model_path']))

        # Define a fixed noise to be used as starting point for generation
        self.fixed_noise = torch.randn(self.config['class_images'], self.config['gen']['latentspace_dim'], 1, 1, device=device)

        # Define a vector to encode the labels
        self.labels = self.label_preprocessing()

        self.folders_preprocess()

    def folders_preprocess(self):
        """
        Creates, if needed, a sub-folder for each class, in order to contain the generated images.
        """

        for idx, class_name in enumerate(self.config['classes']):
            os.makedirs(os.path.join(self.config['generated_images_path'], class_name), exist_ok=True)

    def label_preprocessing(self):
        """
        Label preprocessing for the generator. Basically, are needed onehot vectors in order to encode the labels.

        Returns
        -------
        A vector containing the onehot encoding of each label.
        """

        labels = torch.zeros(self.config['num_label'], self.config['num_label'])
        labels = labels.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(self.config['num_label'], 1), 1)
        labels = labels.view(self.config['num_label'], self.config['num_label'], 1, 1)

        return labels

    def generate(self):
        """
        Function that generates test samples starting from fixed noise.

        Returns
        -------
        The generated images.
        """
        
        self.generator.eval()

        with torch.no_grad():
            # Generate a number 'class_images' of images for each class
            for idx, class_name in enumerate(self.config['classes']):
                label = (torch.ones(self.config['class_images']) * idx).type(torch.LongTensor) # [0,0,0,0,...] -> torch.Size([class_images])
                label_gen = self.labels[label].to(self.device) # -> torch.Size([class_images, 10, 1, 1])
                output = self.generator(self.fixed_noise, label_gen) # -> torch.Size([class_images, 3, image_w, image_h])

                for i, image in enumerate(output):
                    result_path = os.path.join(os.path.join(self.config['generated_images_path'], class_name), f"image_{i + 1}.jpg")
                    vutils.save_image(image.data.cpu(), result_path, nrow=1, padding=0, normalize=True)
                    print(f"Saved generated image into {result_path}...")

        self.generator.train()