import os
import torch

import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from torch.utils.data.dataloader import DataLoader

class Tester(object):
    """
    This class implements the testing for the classifier.
    """

    def __init__(self, test_loader: DataLoader, device: torch.device, args: dict, config: dict):
        """
        Parameters
        ----------
        test_loader : DataLoader
            The DataLoader for the test images.

        args : dict
            The arguments passed as program launch.

        config : dict
            The configuration file.
        """

        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.config = config

        # Load model and unfreeze learning only for Fully Connected layer
        if (self.config['model'] == 'alexnet'):
            self.model = models.alexnet(weights='DEFAULT')
            # Reshape last layer according to the number of classes
            num_feature = self.model.classifier[len(self.model.classifier) - 1].in_features
            self.model.classifier[len(self.model.classifier) - 1] = NN.Linear(num_feature, self.config['num_label'])
        elif (self.config['model'] == 'resnet'):
            self.model = models.resnet152(weights='DEFAULT')
            # Reshape last layer according to the number of classes
            num_feature = self.model.fc.in_features
            self.model.fc = NN.Linear(num_feature, self.config['num_label'])
        else:
            raise Exception(f"Model {self.config['model']} not implemented. Please fix the configuration file.")

        self.model.load_state_dict(torch.load(os.path.join(self.config['model_path'], self.args.model_name)))
        
        self.model.to(self.device)

    def test(self):
        """
        Perform the test over the test set specified.
        """

        # Used for Accuracy
        correct = 0
        total = 0

        # Model into evaluation mode
        self.model.eval()

        # Don't need to calculate the gradients for outputs
        with torch.no_grad():
            for idx, (images, labels) in enumerate(self.test_loader, 0):
                images, labels = images.to(self.device), labels.to(self.device)
                # Running images through the network
                outputs = self.model(images)

                # Class with the highest energy is what we choose as prediction
                predictions = (torch.max(outputs.data, 1)[1])

                # Computing accuracy
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = 100 * correct / total
        
        print(f"Testing network on the 10000 test images:\n\t\t - accuracy = {accuracy} %")

        self.model.train()