import os
import argparse
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader


from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10

def argparser():
    args = argparse.ArgumentParser()

    args.add_argument('--run_name', type=str, help='The name of the run will be used to store the results.', required=True)

    args.add_argument('--image_size', type=int, default=32, help='The resized dimension of the images.', required=False)
    args.add_argument('--num_image_channels', type=int, default=3, help='The number of channels in the inpt images.', required=False)
    args.add_argument('--num_label', type=int, default=10, help='Number of labels.', required=False)
    args.add_argument('--num_workers', type=int, default=2, help='Number of workers in data loader.', required=False)

    args.add_argument('--latentspace_dim', type=int, default=100, help='The dimension of the latent space in the model.', required=False)
    args.add_argument('--gen_featuremap_dim', type=int, default=64, help='Dimension of the feature map in the generator.', required=False)
    args.add_argument('--dis_featuremap_dim', type=int, default=64, help='Dimension of the feature map in the discriminator.', required=False)

    args.add_argument('--epochs', type=int, default=100, help='Number of epochs.', required=False)
    args.add_argument('--batch_size', type=int, default=16, help='Number of elements in batch size.', required=False)

    args.add_argument('--tensorboard_path', type=str, default='../results/Tensorboard/', help='Path to the tensorboard writer.', required=False)
    args.add_argument('--dataset_path', type=str, default='../data/', help='Path were it is located the dataset.', required=False)

    return args.parse_args()


def main(args):
    writer = SummaryWriter(os.path.join(args.tensorboard_path, args.run_name))

    transformList = transforms.Compose([transforms.Resize(args.image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(0.5,))])

    cifar10_dataset = CIFAR10(root=args.dataset_path,
                              train=True,
                              transform=transformList,
                              download=True)

    cifar10_dataloader = DataLoader(dataset=cifar10_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Code will be executed on {device}")
    

if __name__ == "__main__":
    # To suppress tensorflow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    args = argparser()
    main(args)