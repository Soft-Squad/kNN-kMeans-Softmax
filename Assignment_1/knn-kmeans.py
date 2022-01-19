import sys
import pickle
import numpy

def unpickle_cifar10(file):

    # Each of the batch file contains a dictionary with the following elements:
    # data -- 10000x3072 numpy array of uint8S. Each row stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    # labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

def main():
    unpickle_cifar10(sys.argv[1])
    print("Test")

if __name__ == '__main__':
    main()