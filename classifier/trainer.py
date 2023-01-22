import os
import torch

import pandas as pd
import numpy as np
import seaborn as sn

import torch.nn as NN
import torch.optim as optim
import torchvision.models as models

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

class FineTune(object):
    """
    This class implements the training for the pretrained classifier.
    """

    def __init__(self, writer: SummaryWriter, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, args: dict, config: dict):
        """
        Parameters
        ----------
        writer : SummaryWriter
            The SummaryWriter for storing infos about training in a tensorboard format.

        train_loader : DataLoader
            The DataLoader for the train images.

        test_loader : DataLoader
            The DataLoader for the test images.

        args : dict
            The arguments passed as program launch.

        config : dict
            The configuration file.
        """

        self.writer = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.args = args
        self.config = config

        self.model_name = self.args.run_name

        # Load model and unfreeze learning only for Fully Connected layer
        if (self.config['model'] == 'alexnet'):
            self.model = models.alexnet(weights='DEFAULT')
            self.set_parameter_requires_grad(self.model.features, False)
            # Reshape last layer according to the number of classes
            num_feature = self.model.classifier[len(self.model.classifier) - 1].in_features
            self.model.classifier[len(self.model.classifier) - 1] = NN.Linear(num_feature, self.config['num_label'])
        elif (self.config['model'] == 'resnet'):
            self.model = models.resnet152(weights='DEFAULT')
            for idx, (name, module) in enumerate(self.model.named_children(), 0):
                if name != 'fc':
                    self.set_parameter_requires_grad(module, False)
            # Reshape last layer according to the number of classes
            num_feature = self.model.fc.in_features
            self.model.fc = NN.Linear(num_feature, self.config['num_label'])
        else:
            raise Exception(f"Model {self.config['model']} not implemented. Please fix the configuration file.")
        
        self.model.to(device)

        # Define Loss function
        if self.config['loss'] == 'CrossEntropy':
            self.criterion = NN.CrossEntropyLoss()
        else:
            raise Exception(f"{self.config['loss']} not implemented. Please fix the configuration file.")

        # Define Optimizer
        if self.config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], betas=(self.config['beta'], 0.999))
        elif self.config['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        else:
            raise Exception(f"{self.config['optimizer']} not implemented. Please fix the configuration file.") 

    def save_model(self, idx: int):
        """
        Function that saves the model.

        Parameters
        ----------
        idx : int
            The iteration in which the model was saved.
        """

        path = os.path.join(self.config['model_path'], f"{self.model_name}_epoch_{idx}.pth")
        torch.save(self.model.state_dict(), path)
        print("Model saved!")

    def set_parameter_requires_grad(self, model, req_grad=False):
        """"
        Function that freezes the model layers.

        Parameters
        ----------
        req_grad : bool
            Set False if you want to freeze the layer, True otherwise.
        """

        for param in model.parameters():
            param.requires_grad = req_grad

    def test(self, iter: int, epoch: int):
        """
        Perform the test in order to keep trace the performances during the training.

        Parameters
        ----------
        iter : int
            The iteration number.

        epoch : int
            The epoch number.
        """

        predicted = [] # save the prediction
        groundtruth = [] # save the ground truth

        # Model into evaluation mode
        self.model.eval()

        # Don't need to calculate the gradients for outputs
        with torch.no_grad():
            for idx, (image, label) in enumerate(self.test_loader, 0):
                image, label = image.to(self.device), label.to(self.device)
                # Running images through the network
                outputs = self.model(image)

                # Class with the highest energy is what we choose as prediction
                prediction = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
                predicted.extend(prediction)

                label = label.data.cpu().numpy()
                groundtruth.extend(label)

        accuracy = accuracy_score(y_true=groundtruth, y_pred=predicted)
        precision = precision_score(y_true=groundtruth, y_pred=predicted, average='macro') # 'macro' = all classes have the same weight
        recall = recall_score(y_true=groundtruth, y_pred=predicted, average='macro')
        f1 = f1_score(y_true=groundtruth, y_pred=predicted, average='macro')

        sklearn_cm = confusion_matrix(groundtruth, predicted)
        df_cm = pd.DataFrame(sklearn_cm/np.sum(sklearn_cm) * 10, index=[i for i in self.config['classes']], columns=[i for i in self.config['classes']])
        cf_matrix = sn.heatmap(df_cm, annot=True).get_figure()

        self.writer.add_scalar('Test accuracy', accuracy, epoch * len(self.train_loader) + iter)
        self.writer.add_scalar('Test precision', precision, epoch * len(self.train_loader) + iter)
        self.writer.add_scalar('Test recall', recall, epoch * len(self.train_loader) + iter)
        self.writer.add_scalar('Test f1 score', f1, epoch * len(self.train_loader) + iter)

        self.writer.add_figure("Test confusion matrix", cf_matrix, epoch * len(self.train_loader) + iter)
        
        print(f"Testing network on the 10000 test images:\n\t\t - accuracy = {accuracy} %\n\t\t - precision = {precision} %\n\t\t - recall = {recall}\n\t\t - f1 score = {f1}")

        self.model.train()

    def train(self):
        """
        Fine tuning function.
        """

        print("Fine tuning started...")

        for epoch in range(self.config['epochs']):
            running_loss = 0.0

            for idx, (image, label) in enumerate(self.train_loader, 0):
                # Data on correct device
                image, label = image.to(self.device), label.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                output = self.model(image)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Output training stats
                if idx % self.config['print_every'] == self.config['print_every'] - 1:
                    self.writer.add_scalar('training loss', running_loss / self.config['print_every'], epoch * len(self.train_loader) + idx)
                    
                    print(f"[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / self.config['print_every']:.3f}")
                    running_loss = 0.0

                    # Test model
                    self.test(idx, epoch)

            if ((epoch % self.config['save_every'] == self.config['save_every'] - 1) or (epoch == len(self.train_loader) - 1)):
                self.save_model(epoch)

        self.writer.flush()
        self.writer.close()
        print('Finished Fine Tuning')
