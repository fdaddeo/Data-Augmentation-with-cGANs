import os
import torch
import torch.nn as NN
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

import torchvision.utils as vutils

from .asset.generator import Generator
from .asset.discriminator import Discriminator

class Trainer(object):
    """
    This class implements the training for the cGAN.
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

        self.generator_name = f"cgan_gen_{self.args.run_name}.pth"
        self.discriminator_name = f"cgan_dis_{self.args.run_name}.pth"

        # Initialize Models
        self.generator = Generator(config=self.config['gen'],
                                   num_label=self.config['num_label'],
                                   image_channels=self.config['image_channels']).to(self.device)

        self.discriminator = Discriminator(config=self.config['dis'],
                                           num_label=self.config['num_label'],
                                           image_channels=self.config['image_channels']).to(self.device)

        # Define batch of latent vectors that it will be used to visualize the progression of the generator
        self.fixed_noise = torch.randn(8, self.config['gen']['latentspace_dim'], 1, 1, device=device)

        # Apply the weight initialization according to the paper
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        # Define Loss function
        if self.config['loss'] == 'BCE':
            self.criterion = NN.BCELoss()
        elif self.config['loss'] == 'WAS':
            self.criterion = "WAS"

        # Define Optimizer
        if self.config['optimizer'] == 'Adam':
            self.optimizerG = optim.Adam(self.generator.parameters(), lr=self.config['learning_rate'], betas=(self.config['beta'], 0.999))
            self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.config['learning_rate'], betas=(self.config['beta'], 0.999))
        elif self.config['optimizer'] == 'SGD':
            self.optimizerG = optim.SGD(self.generator.parameters(), lr=self.config['learning_rate'], momentum=0.9)
            self.optimizerD = optim.SGD(self.discriminator.parameters(), lr=self.config['learning_rate'], momentum=0.9)

        # Perform label preprocessing
        self.gen_labels = self.gen_label_preprocessing()
        self.dis_labels = self.dis_label_preprocessing()

    def save_models(self):
        """
        Function that saves the models.
        """

        pathG = os.path.join(self.config['gen_checkpoint_path'], self.generator_name)
        pathD = os.path.join(self.config['dis_checkpoint_path'], self.discriminator_name)
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

    def gen_label_preprocessing(self):
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
        
    def dis_label_preprocessing(self):
        """
        Label preprocessing for the discriminator. Basically, are needed onehot "images".

        Returns
        -------
        A vector containing each onehot images encoded for each label.
        """

        images = torch.zeros([self.config['num_label'], self.config['num_label'], self.config['image_size'], self.config['image_size']])
        for i in range(self.config['num_label']):
            images[i, i, :, :] = 1

        return images

    def generate_test(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Function that generates test samples.

        Returns
        -------
        The generated image.
        """
        
        self.generator.eval()

        # label 0
        label = (torch.ones(8) * 0).type(torch.LongTensor) #[0,0,0,0,0,0,0,0]
        label_gen = self.gen_labels[label].to(self.device)
        output = self.generator(noise, label_gen)
        inference_res = output

        # labels 1-9    
        for i in range(1, self.config['num_label']):
            label = (torch.ones(8) * i).type(torch.LongTensor)
            label_gen = self.gen_labels[label].to(self.device)
            output = self.generator(noise, label_gen)
            inference_res = torch.cat([inference_res, output], dim = 0)
        
        self.generator.train()

        return inference_res

    def train(self):
        """
        Implements the training of the network.
        """

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")

        for epoch in range(self.config['epochs']):
            for idx, (image, label) in enumerate(self.train_loader, 0):
                ############################
                # (1) Update Discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                ## First train with all-real batch
                self.discriminator.zero_grad()
                batch_size = image.size(0)

                # Establish convention for real and fake labels during training
                real_label = torch.ones(batch_size).to(self.device)
                fake_label = torch.zeros(batch_size).to(self.device)

                # Format batch
                real_image = image.to(self.device)
                real_label_dis = self.dis_labels[label].to(self.device)

                # Forward pass real batch through Discriminator
                output = self.discriminator(real_image, real_label_dis).view(-1)

                # Calculate loss on all-real batch
                error_dis_real = self.criterion(output, real_label)

                # Calculate gradients for Discriminator in backward pass
                error_dis_real.backward()
                discriminator_x = output.mean().item()

                ## Then train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, self.config['dis']['latentspace_dim'], 1, 1, device=self.device)

                # Pick random label ang generate corresponding onehot
                random_label = (torch.rand(batch_size) * self.config['num_label']).type(torch.LongTensor) # equivalent to int64 #[0,6,4,3,9]
                random_label_gen = self.gen_labels[random_label].to(self.device)

                # Generate fake image batch with G
                fake_image = self.generator(noise, random_label_gen)
                
                # Classify all-fake batch with Discriminator
                random_label_dis = self.dis_labels[random_label].to(self.device)
                output = self.discriminator(fake_image.detach(), random_label_dis).view(-1)

                # Calculate Discriminator's loss on the all-fake batch
                error_dis_fake = self.criterion(output, fake_label)

                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                error_dis_fake.backward()
                discriminator_generator_z1 = output.mean().item()

                # Compute error of Discriminator as sum over the fake and the real batches
                error_dis_tot = error_dis_real + error_dis_fake

                # Update Discriminator
                self.optimizerD.step()

                ############################
                # (2) Update Generator network: maximize log(D(G(z)))
                ###########################

                self.generator.zero_grad()

                # Since we just updated the Discriminator, perform another forward pass of all-fake batch through Discriminator
                output = self.discriminator(fake_image, random_label_dis).view(-1)

                # Calculate G's loss based on this output
                error_gen = self.criterion(output, real_label) # fake images are real for generator cost

                # Calculate gradients for Generator
                error_gen.backward()
                discriminator_generator_z2 = output.mean().item()

                # Update Generator
                self.optimizerG.step()

                # Output training stats
                if idx % self.config['print_every'] == self.config['print_every'] - 1:
                    self.writer.add_scalar('generator loss', error_gen.item(), epoch * len(self.train_loader) + idx)
                    self.writer.add_scalar('discriminator loss', error_dis_tot.item(), epoch * len(self.train_loader) + idx)

                    print(f"[{epoch}/{self.config['epochs']}][{idx}/{len(self.train_loader)}]\tLoss_D: {error_dis_tot.item()}\tLoss_G: {error_gen.item()}\tD(x): {discriminator_x}\tD(G(z)): {discriminator_generator_z1} / {discriminator_generator_z2}")

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 250 == 0) or ((epoch == self.config['epochs'] - 1) and (idx == len(self.train_loader) - 1)):
                    with torch.no_grad():
                        fake = self.generate_test(self.fixed_noise).detach().cpu()
                
                    im_grid = vutils.make_grid(fake, padding=2, normalize=True)
                    img_list.append(im_grid)
                    vutils.save_image(im_grid, os.path.join(self.config['generated_images_path'], f"{epoch}_{iters}.jpg"))

                iters += 1
            
            if ((epoch % self.config['save_every'] == 0) or (epoch == self.config['epochs'] - 1)):
                self.save_models()

        # TODO: implement early stopping and ReduceLROnPlateau

        self.writer.flush()
        self.writer.close()
        print("Finished Training")