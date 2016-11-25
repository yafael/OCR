# Author: Kyla Bouldin
# Description: creates training data file based on arial letter characters
# (10x15 image)


import numpy as np
import cv2
from matplotlib import pyplot as plt
	
def createDigitTrainingData():	
	# import training data
	img = cv2.imread('digits.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	# split the image to 5000 cells, each 20x20 size
	cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
	# Make it into a (50, 100, 20, 20) numpy array
	x = np.array(cells)
	# prepare data
	train = x[:,:].reshape(-1,400).astype(np.float32) # Size = (2500,400) (left half)
	#test = x[:,75:100].reshape(-1,400).astype(np.float32) # Size = (2500,400) (right half)

	# create labels for train data
	k = np.arange(10)
	train_labels = np.repeat(k,500)[:,np.newaxis]
	#test_labels = np.repeat(k,125)[:,np.newaxis]
		
	#initiate kNN, train the data
	knn = cv2.KNearest()
	knn.train(train, train_labels)

	# test data
# 	ret,result,neighbours,dist = knn.find_nearest(test,k=5)
# 	matches = result==test_labels
# 	correct = np.count_nonzero(matches)
# 	accuracy = correct*100.0/result.size
# 	print accuracy
   
	# import and reshape testing data
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
	 	
def createLetterTrainingData():	
	# import training data
	img = cv2.imread('letters.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	# split the image to 130 cells, each 20x20 size
	cells = [np.hsplit(row,26) for row in np.vsplit(img,5)]
	# Make it into a (5, 26, 20, 20) numpy array
	x = np.array(cells)
	# prepare data
	train = x[:,:].reshape(-1,400).astype(np.float32) # Size = (130,400)

	# create labels for train data
	k = np.arange(26)
	train_labels = np.repeat(k,5)[:,np.newaxis]
		
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
		
createDigitTrainingData()
createLetterTrainingData()