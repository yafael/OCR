# Author: Kyla Bouldin, Yashvardhan Gusani
# Description: Tests the trained data on test images

import cv2
import numpy as np
import operator
import os

# Constants
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# ====================
# Custom function designed to sort the contours
# ====================
def sortContours(point1):
    tolerance_factor = 60
    mom1 = cv2.moments(point1)

    x = int(mom1['m10'] / mom1['m00'])
    y = int(mom1['m01'] / mom1['m00'])

    return ((y // tolerance_factor) * tolerance_factor) * 100 + x


# ====================
# takes in image and finds all contours and prints out words it finds
# ====================
def findContours(testFileName):
	# try to load classifications
	try:
		classificationLabels = np.loadtxt("classification_labels.txt", np.float32)
	except:
		print "error, unable to open classification labels, exiting/n"
		os.system("pause")
		return

	# try to load training data	
	try:
		trainingData = np.loadtxt("training_data.txt", np.float32)
	except:
		print "error, unable to open training data, exiting\n"
		os.system("pause")
		return
			
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
	contours.sort(key=lambda x: sortContours(x))
	
	
	
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
	cv2.imshow("imgTestingNumbers", testImage)
	cv2.waitKey(0)

	cv2.destroyAllWindows()
			
	return finalString


findContours("testimage.png")