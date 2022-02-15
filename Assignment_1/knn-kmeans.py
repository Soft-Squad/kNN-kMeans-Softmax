import warnings

from torchvision import datasets  # Contains the CIFAR-10 Dataset
from torchvision.transforms import ToTensor  # Tensor is used to encode the inputs and outputs
import numpy as np
import matplotlib.pyplot as plt     # Used to help visualize data
import random


max_iter = 300

# kNN Algorithm:
# 1. Choose the value of K
# 2. For each point in test data:
#   - Find the Euclidean distance to all the training data points
#   - Store the Euclidean distances in a list and sort it
#   - Choose the first k points
#   - Assign a class to the test point based on the majority of classes present in the chosen points


def compute_distances(X_Train, X):
    # Distances formula from: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    distances = -2 * np.dot(X, X_Train.T) + np.sum(X_Train**2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
    return distances


def knn_predict(y_train, dists, k=3):
    # print(k)
    num_test = dists.shape[0]
    # print(num_test)
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        nearest_neighbor_ids = np.argsort(dists[i])
        closest_y = list(y_train[nearest_neighbor_ids[:k]])
        pass
        y_pred[i] = (np.argmax(np.bincount(closest_y)))
        pass
    return y_pred


def knn_accuracy(y_prediction, y_test, num_test):
    correct = np.sum(y_prediction == y_test)
    accuracy = (float(correct) / num_test) * 100
    #print("Correct: %d/%d\nAccuracy: %f" % (correct, num_test, accuracy))
    return accuracy


def kNN_cross_validation(X_train, X_test, y_train, y_test, num_test):
    folds = 5
    k_list = [3, 5, 7, 11]

    X_train_folds = []
    y_train_folds = []

    X_train_folds = np.array_split(X_train, folds)
    y_train_folds = np.array_split(y_train, folds)
    k_accuracy = {}

    for k in k_list:
        k_accuracy[k] = []
        for k_num in range(0, folds):
            X_test = X_train_folds[k_num]
            y_test = y_train_folds[k_num]
            X_train = X_train_folds
            y_train = y_train_folds

            tmp = np.delete(X_train, k_num, axis=0)
            X_train = np.concatenate((tmp), axis=0)
            y_train = np.delete(y_train, k_num, axis=0)
            y_train = np.concatenate((y_train), axis=0)

            distances = compute_distances(X_train, X_test)
            prediction = knn_predict(y_train, distances, k)

            accuracy = knn_accuracy(prediction, y_test, num_test)
            k_accuracy[k].append(accuracy)

    print("5-Fold Accuracies for k: \n")
    for k in sorted(k_accuracy):
        for acc in k_accuracy[k]:
            print("k = %d, accuracy = %f" % (k, acc))


# KMeans Algorithm:
# 1. Specify number of cluster's k.
# 2. Initialize k points (corresponding to k clusters) randomly from the data. We call these points centroids.
# 3. For each data point, measure the L2 distance from the centroid.
#    Assign each data point to the centroid for which it has the shortest distance.
#    In other words, assign the closest centroid to each data point.
# 4. Now each data point assigned to a centroid forms an individual cluster.
#    For k centroids, we will have k clusters.
#    Update the value of the centroid of each cluster by the mean of all the data points present in that particular cluster.
# 5. Repeat steps 2-4 until the maximum change in centroids for each iteration falls below a threshold value,
#    or the clustering error converges.


def init_centroids(X, k):
    centroids = X[random.sample(range(X.shape[0]), k)]
    return centroids


def compute_centroid_dist(X, centroid):
    dist = ((X - centroid) ** 2).sum(axis=X.ndim - 1)
    return dist


def closet_centroid(X, centroids):
    dist = compute_centroid_dist(X, centroids)
    return np.argmin(dist, axis=1)


def compute_sse(X, centroids, a_centroids):
    sse = 0
    sse = compute_centroid_dist(X, centroids[a_centroids]).sum() / len(X)
    return sse


def kmeans_fit(X_train, X, centroids, k=3):
    np.random.shuffle(X_train)      # Shuffles Data
    random.seed(0)                  # Set for reproducibility

    # centroids = init_centroids(X_train, k)
    a_centroids = np.zeros(len(X_train), dtype=np.int32)

    sse_list = []

    for n in range(50):
        a_centroids = closet_centroid(X_train[:, None, :], centroids[None, :, :])
        for c in range(k):
            cluster = X_train[a_centroids == c]
            cluster = cluster.mean(axis=0)

            centroids[c] = cluster

        sse = compute_sse(X_train.squeeze(), centroids.squeeze(), a_centroids)
        sse_list.append(sse)

    kmeans_y_pred = closet_centroid(X[:, None, :], centroids[None, :, :])
    return kmeans_y_pred


def kmeans_accuracy(y_pred, y_test, num_test):
    correct = np.sum(y_pred == y_test)
    accuracy = (float(correct) / num_test) * 100
    #print("Correct: %d/%d\nAccuracy: %f" % (correct, num_test, accuracy))
    return accuracy


def kmeans_cross_validation(X_train, X_test, y_test, num_test):
    folds = 5
    k_list = [3, 5, 7, 11]

    X_train_folds = []

    X_train_folds = np.array_split(X_train, folds)
    k_accuracy = {}

    for k in k_list:
        centroids = init_centroids(X_train, k)
        k_accuracy[k] = []
        for k_num in range(0, folds):
            X_test = X_train_folds[k_num]
            X_train = X_train_folds

            tmp = np.delete(X_train, k_num, axis=0)
            X_train = np.concatenate((tmp), axis=0)

            prediction = kmeans_fit(X_train, X_test, centroids, k)

            accuracy = kmeans_accuracy(prediction, y_test, num_test)
            k_accuracy[k].append(accuracy)

    print("5-Fold Accuracies for k: \n")
    for k in sorted(k_accuracy):
        for acc in k_accuracy[k]:
            print("k = %d, accuracy = %f" % (k, acc))


def main():
    # Download CIFAR-10 to a folder called 'data'
    train_dataset = datasets.CIFAR10(root='data/', download=True, train=True, transform=ToTensor())
    test_dataset = datasets.CIFAR10(root='data/', download=True, train=False, transform=ToTensor())

    np.set_printoptions(threshold=np.inf)   # Allows printing of 50000 items

    # Get training data (n = 50000) and labels from the dataset
    X_train = train_dataset.data
    y_train = np.array(train_dataset.targets)
    X_train_reshaped = X_train.reshape((50000, 32 * 32 * 3)).astype('float')  # In the form of a np.array
    # Get test data (n = 10000) and labels from the dataset
    X_test = test_dataset.data
    y_test = np.array(test_dataset.targets)
    X_test_reshaped = X_test.reshape((10000, 32 * 32 * 3)).astype('float')  # In the form of a np.array

    num_test = X_test_reshaped.shape[0]

    classes = train_dataset.classes     # Image class names from CIFAR-10

    # distances = compute_distances(X_train_reshaped, X_test_reshaped)
    # kNN_cross_validation(X_train_reshaped, X_test_reshaped, y_train, y_test, num_test)
    # y_prediction = knn_predict(y_train, distances, k=11)

    # Print the accuracy results of our kNN implementation
    #knn_accuracy(y_prediction, y_test, num_test)

    # Start testing K-Means
    kmeans_cross_validation(X_train_reshaped, X_test_reshaped, y_test, num_test)
    # kmean_prediction = kmeans_fit(X_train_reshaped, X_test_reshaped, k=3)
    # kmeans_accuracy(kmean_prediction, y_test, num_test)


if __name__ == '__main__':
    main()
