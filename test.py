# Author: Kyla Bouldin, Yashvardhan Gusani
# Description: Tests the trained data on test images

import cv2
import numpy as np
import operator
import os
import sortContours as sort

# Constants
MIN_CONTOUR_AREA = 10

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# ====================
# takes in image and finds all contours and prints out words it finds
# ====================
def findContours(testFileName):
	# load classifications and test data
	classificationLabels = np.loadtxt("classification_labels.txt", np.float32)
	trainingData = np.loadtxt("training_data.txt", np.float32)
	
	# reshape classifications to 1d
	classificationLabels = classificationLabels.reshape((classificationLabels.size, 1))
	
	# create KNearest and train
	kNearest = cv2.KNearest()
	kNearest.train(trainingData, classificationLabels)
	
	
	# declare empty lists
	contours = []
	
	testImage = cv2.imread(testFileName)
	testImGray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
	testImBlurred = cv2.GaussianBlur(testImGray, (5, 5), 0)
	# Filter image from grayscale to black and white
	imgThresh = cv2.adaptiveThreshold(testImBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	# make a copy
	imgThreshCopy = imgThresh.copy() 
	
	contours, heierachy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key=lambda x: sort.sortContours(x))
	
	
	finalString = ""
	
	for contour in contours:
		[intX, intY, intWidth, intHeight] = cv2.boundingRect(contour)
		area = cv2.contourArea(contour) 
		
		if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
			cv2.rectangle(testImage,(intX, intY),(intX + intWidth, intY + intHeight), (0, 255, 0), 2)
			# crop image
			letter = imgThresh[intY : intY + intHeight, intX : intX + intWidth]
												 
			letter = cv2.resize(letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
			letter = letter.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
			letter = np.float32(letter)
			
			ret, result, neighbors, dist = kNearest.find_nearest(letter, k=1)
			
			currentChar = str(chr(int(ret)))
			
			# append current char to full string
			finalString = finalString + currentChar
			
	print "\n" + finalString + "\n"
	cv2.imshow("testImage", testImage)
	cv2.waitKey(0)

	cv2.destroyAllWindows()
			
	return finalString


findContours("testdata/courier_test.png")