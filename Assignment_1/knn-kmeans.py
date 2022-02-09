from torchvision import datasets  # Contains the CIFAR-10 Dataset
from torchvision.transforms import ToTensor  # Tensor is used to encode the inputs and outputs
import numpy as np
import matplotlib.pyplot as plt     # Used to help visualize data


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
    # print("Correct: %d/%d\nAccuracy: %f" % (correct, num_test, accuracy))
    return accuracy

# KMeans Algorithm:
# 1. Specify number of cluster's K.
# 2. Initialize centroids by first shuffling the dataset
#    and then randomly selecting K data points for the centroids without replacement
# 3. Keep iterating until there is no change to the centroids.
#    i.e. assignment of data points to cluster's isn't changing.
#   - Compute the sum of the squared distance between data points and all centroids.
#   - Assign each data point to the closest cluster (centroid).
#   - Compute the centroids for the clusters by taking the average of all the data points that belong to each cluster.


def init_centroids(X, random_state, num_clusters):
    np.random.RandomState(random_state)
    rand_idx = np.random.permutation(X.shape[0])
    centroids = X[rand_idx[:num_clusters]]
    return centroids


def centroid_distances(X, centroids, num_clusters):

    distance = np.zeros((X.shape[0], num_clusters))
    for k in range(num_clusters):
        row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
        distance[:, k] = np.square(row_norm)
    return distance


def compute_centroids(X, num_clusters, labels):
    centroids = np.zeros((num_clusters, X.shape[1]))
    for i in range(num_clusters):
        centroids[i, :] = np.mean(X[labels == i, :], axis=0)
    return centroids


def closest_cluster(dist):
    return np.argmin(dist, axis=1)


def fit_kmeans(X, X_Train,  k=7):
    num_cluster = k
    random_state = 123
    centroids = init_centroids(X_Train, random_state, num_cluster)
    #print(centroids)
    for i in range(100):
        old_centroids = centroids
        distance = centroid_distances(X_Train, old_centroids, num_cluster)
        #print(distance)
        labels = closest_cluster(distance)
        #print(labels)
        centroids = compute_centroids(X_Train, num_cluster, labels)
        if np.all(old_centroids == centroids):
            break

    kmean_prediction = kmeans_prediction(X, centroids, num_cluster)

    return kmean_prediction


def kmeans_prediction(X, centroids, num_cluster):
    distance = centroid_distances(X, centroids, num_cluster)
    return closest_cluster(distance)


def kmeans_accuracy(kmeans_prediction, y_test, num_test):
    correct = np.sum(kmeans_prediction == y_test)
    accuracy = (float(correct) / num_test) * 100
    print("Correct: %d/%d\nAccuracy: %f" % (correct, num_test, accuracy))


def cross_validation(X_train, X_test, y_train, y_test, num_test):
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
    cross_validation(X_train_reshaped, X_test_reshaped, y_train, y_test, num_test)
    # y_prediction = knn_predict(y_train, distances, k=11)

    # Print the accuracy results of our kNN implementation
    # knn_accuracy(y_prediction, y_test, num_test)

    # Start testing K-Means
    # squared_norms = np.square(distances)        # Could be useful to compute centroids distances
    # kmean_prediction = fit_kmeans(X_test_reshaped, X_train_reshaped, k=3)

    # kmeans_accuracy(kmean_prediction, y_test, num_test)


if __name__ == '__main__':
    main()
