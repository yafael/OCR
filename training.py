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
	cells = [np.hsplit(row,26) for row in np.vsplit(img,5)]
	x = np.array(cells)
	train = x[:,:].reshape(-1,400).astype(np.float32)
	k = np.arange(26)
	train_labels = np.repeat(k,5)[:,np.newaxis]
	np.savez('knn_letter_data.npz',train=train, train_labels=train_labels)
	
# ====================
# use saved data to test alphabet
# ====================
def testLetters():
	with np.load('knn_letter_data.npz') as data:
		train = data['train']
		train_labels = data['train_labels']
		
		#initiate kNN, train the data
		knn = cv2.KNearest()
		knn.train(train, train_labels)
		
	alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	print("Testing Alphabet");
	for i in range(len(alphabet)):
		filename = 'testingdata/%s.png' % alphabet[i]		
		testimg = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)	
		testing = testimg[:,:].reshape(-1, 400).astype(np.float32) # 400 = 20x20	
		test_label = i
	
		# test data for k = 1
		ret,result,neighbors,dist = knn.find_nearest(testing,k=5)
	 	print test_label
	 	print result

# ====================
# given 20x20 grayscale image return estimated letter (in int value)
# ====================
def getCorrespondingLetter(image):
	with np.load('knn_letter_data.npz') as data:
		train = data['train']
		train_labels = data['train_labels']

	knn = cv2.KNearest()
	knn.train(train, train_labels)
	
	testimg = image[:,:].reshape(-1, 400).astype(np.float32)
	ret, result, neighbors, dist = knn.find_nearest(testimg, k=5)
	return result


# ====================
# use opencv digit data to create training set for digits
# ====================	
def createDigitTrainingData():	
	img = cv2.imread('digits.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
	x = np.array(cells)
	train = x[:,:].reshape(-1,400).astype(np.float32) # Size = (2500,400)
	#test = x[:,75:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

	k = np.arange(10)
	train_labels = np.repeat(k,500)[:,np.newaxis]
	#test_labels = np.repeat(k,125)[:,np.newaxis]

	np.savez('knn_digit_data.npz',train=train, train_labels=train_labels)
	# np.savez('knn_digit_test_data.npz',test=train, test_labels=test_labels)


# ====================
# use saved data to test digits
# ====================
def testDigits():
	with np.load('knn_digit_data.npz') as data:
		train = data['train']
		train_labels = data['train_labels']
	
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
		testing = testimg[:,:].reshape(-1, 400).astype(np.float32) # 400 = 20x20	
		test_label = i
	
		# test data for k = 1
		ret,result,neighbors,dist = knn.find_nearest(testing,k=5)
	 	print test_label
	 	print result
	 	
generateLetterTrainingData()	 	
testLetters()
createDigitTrainingData()
testDigits()

testaimage = cv2.imread('testingdata/a.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)	
print(getCorrespondingLetter(testaimage))