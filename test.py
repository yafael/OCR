# Author: Kyla Bouldin, Yashvardhan Gusani
# Description: Tests the trained data on test images

import cv2
import numpy as np
import operator
import os
import sortContours as sort
import math

# Constants
MIN_CONTOUR_AREA = 10

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def getDistanceBetween(charA, charB):
	MomCharA = cv2.moments(charA)
	MomCharB = cv2.moments(charB)
	cXA = int(MomCharA["m10"] / MomCharA["m00"])
	cYA = int(MomCharA["m01"] / MomCharA["m00"])
	cXB = int(MomCharB["m10"] / MomCharB["m00"])
	cYB = int(MomCharB["m01"] / MomCharB["m00"])
	intX = abs(cXA - cXB)
	intY = abs(cYA - cYB)
	return math.sqrt((intX ** 2) + (intY ** 2))

# ====================
# takes in image and finds all contours and prints out words it finds
# ====================
def findContours(testFileName):
	# load classifications and test data
	try:
		classificationLabels = np.loadtxt("classification_labels.txt", np.float32)
	except:
		print("Can't find classification labels")
	
	try:
		trainingData = np.loadtxt("training_data.txt", np.float32)
	except:
		print("Can't find training data")
		
	print(len(classificationLabels))
	print(len(trainingData))
	
	# reshape classifications to 1d
	classificationLabels = classificationLabels.reshape((classificationLabels.size, 1))
	
	# create KNearest and train
	kNearest = cv2.KNearest()
	kNearest.train(trainingData, classificationLabels)
		
	testImage = cv2.imread(testFileName)
	testImGray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
	imgThresh = cv2.adaptiveThreshold(testImGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	imgThreshCopy = imgThresh.copy() 
	
	contours, heierachy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key=lambda x: sort.sortContours(x))

	
	finalString = ""
	characterContourList = []
	
	for contour in contours:
		area = cv2.contourArea(contour) 
		
		if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
			characterContourList.append(contour)
	
	# get mean distance between chars
	nextChar = ""
	distanceList = []
	for i in range(len(characterContourList) - 1):
		character = characterContourList[i]
		nextChar = characterContourList[i + 1]
		distance = getDistanceBetween(nextChar, character)
		distanceList.append(distance)
	
	meanDistance = np.mean(distanceList)
	stdDev = np.std(distanceList)
	
	nextChar = ""
	for i in range(len(characterContourList)):
		character = characterContourList[i]
		[intX, intY, intWidth, intHeight] = cv2.boundingRect(character)
		cv2.rectangle(testImage,(intX, intY),(intX + intWidth, intY + intHeight), (0, 255, 0), 2)
		# crop image
		letter = imgThresh[intY : intY + intHeight, intX : intX + intWidth]
											 
		letter = cv2.resize(letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
		
		# make 1d
		letter = letter.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
		letter = np.float32(letter)
		
		ret, result, neighbors, dist = kNearest.find_nearest(letter, k=2)
		
		currentChar = str(chr(int(ret)))
		finalString = finalString + currentChar
		
		# detect space
		if (i < len(characterContourList) - 1):
			nextChar = characterContourList[i + 1]
			distance = getDistanceBetween(nextChar, character)
			if (abs(distance - meanDistance) > stdDev):
				finalString = finalString + " "
			
	print "\n" + finalString + "\n"
	cv2.imshow("testImage", testImage)
	cv2.waitKey(0)

	cv2.destroyAllWindows()
			
	return finalString


findContours("testdata/couriernew_test.png")