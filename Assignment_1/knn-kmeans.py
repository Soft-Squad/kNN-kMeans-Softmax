import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # Contains the CIFAR-10 Dataset
from torchvision.transforms import ToTensor, Compose  # Tensor is used to encode the inputs and outputs
import numpy as np
import matplotlib.pyplot as plt     # Used to help visualize data
import scipy.stats


def compute_distances(X_Train, X):
    # Distances formula from: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    distances = -2 * np.dot(X, X_Train.T) + np.sum(X_Train**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
    return distances


def y_predict(y_train, dists, k=1):
    #print(k)
    num_test = dists.shape[0]
    #print(num_test)
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        nearest_neighbor_ids = np.argsort(dists[i])
        closest_y = list(y_train[nearest_neighbor_ids[:k]])
        pass
        y_pred[i] = (np.argmax(np.bincount(closest_y)))
        pass
    return y_pred


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

    num_test = x_test_reshaped.shape[0]

    classes = train_dataset.classes     # Image class names from CIFAR-10

    distances = compute_distances(x_train_reshaped, x_test_reshaped)
    y_prediction = y_predict(y_train, distances, k=11)

    correct = np.sum(y_prediction == y_test)
    accuracy = (float(correct) / num_test) * 100
    print("Correct: %d/%d\nAccuracy: %f" % (correct, num_test, accuracy))


if __name__ == '__main__':
    main()
