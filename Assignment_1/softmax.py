from torchvision import datasets  # Contains the CIFAR-10 Dataset
from torchvision.transforms import ToTensor  # Tensor is used to encode the inputs and outputs
import numpy as np
import matplotlib.pyplot as plt     # Used to help visualize data


def CIFAR10_data(train_dataset, test_dataset):
    X_train = train_dataset.data
    y_train = np.array(train_dataset.targets)
    X_test = test_dataset.data
    y_test = np.array(test_dataset.targets)

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype('float')
    X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype('float')

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # Add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

    return X_train, y_train, X_test, y_test


def softmax(X, X_test, y, y_test, regression):
    W = np.random.randn(10, 3073) * 0.0001
    loss, grad = softmax_loss(W, X, y, regression)

    loss_record = softmax_fit(W, X, y, lr=1e-6, regression=1e-4)

    y_test_pred = softmax_predict(W, X_test)

    num_test = X_test.shape[1]
    softmax_accuracy(y_test_pred, y_test, num_test)


def softmax_fit(W, X, y, lr=1e-5, regression=1e-3):
    _, N = X.shape
    loss_history = []
    for i in range(1000):
        idx = np.random.choice(N, 256, replace=True)
        X_batch = X[:, idx]
        y_batch = y[idx]

        loss, grad = softmax_loss(W, X_batch, y_batch, regression)
        loss_history.append(loss)

        W -= lr * grad

    return loss_history


def softmax_loss(W, X, y, regression):
    f = np.dot(W, X)
    f -= np.max(f, axis=0)
    P = np.exp(f) / np.sum(np.exp(f), axis=0)
    L = -1 / len(y) * np.sum(np.log(P[y, range(len(y))]))
    R = 0.5 * np.sum(np.multiply(W, W))

    loss = L + R * regression

    P[y, range(len(y))] -= 1
    dW = 1 / len(y) * P.dot(X.T) + regression * W

    return loss, dW


def softmax_predict(W, X):
    y = W.dot(X)
    y_pred = np.argmax(y, axis=0)
    return y_pred


def softmax_accuracy(y_pred, y_test, num_test):
    print(y_test.shape)
    print(y_pred.shape)
    correct = np.sum(y_test == y_pred)
    accuracy = (float(correct) / num_test) * 100
    print("Correct: %d/%d\nAccuracy: %f" % (correct, num_test, accuracy))


def main():
    # Download CIFAR-10 to a folder called 'data'
    train_dataset = datasets.CIFAR10(root='data/', download=True, train=True, transform=ToTensor())
    test_dataset = datasets.CIFAR10(root='data/', download=True, train=False, transform=ToTensor())

    np.set_printoptions(threshold=np.inf)  # Allows printing of 50000 items

    X_train, y_train, X_test, y_test = CIFAR10_data(train_dataset, test_dataset)
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    softmax(X_train, X_test, y_train, y_test, 1e-5)


if __name__ == '__main__':
    main()
