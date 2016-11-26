# Author: Kyla Bouldin
# Description: creates training data file based on arial letter characters
# (10x15 image)


import numpy as np
import cv2
from matplotlib import pyplot as plt


# ====================
# use digital letters to create and save training data
# ====================
def generateLetterTrainingData():
    img = cv2.imread('letters.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # Split image into 130 cells (5 x 26) where each cell is 20x20
    cells = [np.hsplit(row, 26) for row in np.vsplit(img, 5)]
    # Turn cells into Numpy array
    x = np.array(cells)
    # Prepare array for training data
    train = x[:, :].reshape(-1, 400).astype(np.float32)
    # Create 26 labels for training data
    k = np.arange(26)
    # Saves training data array as .npz file
    train_labels = np.repeat(k, 5)[:, np.newaxis]
    np.savez('knn_letter_data.npz', train=train, train_labels=train_labels)


# ====================
# use saved data to test alphabet
# ====================
def testLetters():
    # Load training data from .npz file
    with np.load('knn_letter_data.npz') as data:
        train = data['train']
        train_labels = data['train_labels']

        # Use kNN to train the data
        knn = cv2.KNearest()
        knn.train(train, train_labels)

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    print("Testing Alphabet");
    for i in range(len(alphabet)):
        filename = 'testingdata/%s.png' % alphabet[i]
        testimg = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        testing = testimg[:, :].reshape(-1, 400).astype(np.float32)  # 400 = 20x20
        test_label = i

        # Use the trained data for kNN to test data for k = 1
        ret, result, neighbors, dist = knn.find_nearest(testing, k=5)
        print test_label
        print result


# ====================
# given 20x20 grayscale image return estimated letter (in int value)
# ====================
def getCorrespondingLetter(image):
    # Load trained letter data
    with np.load('knn_letter_data.npz') as data:
        train = data['train']
        train_labels = data['train_labels']

    # Use kNN to train data
    knn = cv2.KNearest()
    knn.train(train, train_labels)

    # NOTE(codrienne): I renamed testimg to testing b/c I figured it was a typo.
    testing = image[:, :].reshape(-1, 400).astype(np.float32)
    ret, result, neighbors, dist = knn.find_nearest(testing, k=5)
    return result


# ====================
# use opencv digit data to create training set for digits
# ====================
def createDigitTrainingData():
    img = cv2.imread('digits.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    # Split image into 5000 cells (50 x 100) where each cell is 20x20
    cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
    # Create a Numpy array for cells
    x = np.array(cells)
    # Prepare array for training data
    train = x[:, :].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
        # test = x[:,75:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
    # Create 10 labels
    k = np.arange(10)
    train_labels = np.repeat(k, 500)[:, np.newaxis]
        # test_labels = np.repeat(k,125)[:,np.newaxis]
    # Save training data as .npz file
    np.savez('knn_digit_data.npz', train=train, train_labels=train_labels)
        # np.savez('knn_digit_test_data.npz',test=train, test_labels=test_labels)


# ====================
# use saved data to test digits
# ====================
def testDigits():
    with np.load('knn_digit_data.npz') as data:
        train = data['train']
        train_labels = data['train_labels']
    # Train KNN on training digit data
    knn = cv2.KNearest()
    knn.train(train, train_labels)

    '''
    with np.load('knn_digit_test_data.npz') as data:
        test = data['test']
        test_labels = data['test_labels']

    ret,result,neighbours,dist = knn.find_nearest(test,k=5)
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print accuracy
    '''

    print("Testing Digits");
    for i in range(10):
        filename = 'testingdata/%s.png' % i
        testimg = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        testing = testimg[:, :].reshape(-1, 400).astype(np.float32)  # 400 = 20x20
        test_label = i

        # test data for k = 1
        ret, result, neighbors, dist = knn.find_nearest(testing, k=5)
        print test_label
        print result


generateLetterTrainingData()
testLetters()
createDigitTrainingData()
testDigits()