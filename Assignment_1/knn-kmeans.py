import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # Contains the CIFAR-10 Dataset
from torchvision.transforms import ToTensor, Compose  # Tensor is used to encode the inputs and outputs
import numpy as np
import matplotlib.pyplot as plt     # Used to help visualize data


def knn(X):
    k = 3
    distances = np.linalg.norm(X[0] - X[1])
    print(distances)
    return None


def main():
    # Download CIFAR-10 to a folder called 'data'
    train_dataset = datasets.CIFAR10(root='data/', download=True, train=True, transform=ToTensor())
    test_dataset = datasets.CIFAR10(root='data/', download=True, train=False, transform=ToTensor())

    np.set_printoptions(threshold=np.inf)   # Allows printing of 50000 items

    # Get data and labels from the dataset
    x_train = train_dataset.data
    y_train = np.array(train_dataset.targets)
    x_train_reshaped = x_train.reshape((50000, 32 * 32 * 3)).astype('float')  # In the form of a np.array

    x_test = test_dataset.data
    y_test = np.array(test_dataset.targets)
    x_test_reshaped = x_test.reshape((10000, 32 * 32 * 3)).astype('float')  # In the form of a np.array

    classes = train_dataset.classes     # Image class names from CIFAR-10

    knn(x_train_reshaped)


if __name__ == '__main__':
    main()
