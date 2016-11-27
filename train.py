# Author: Kyla Bouldin, Yashvardhan Gusani
# Description: creates training data and classification files

import sys
import numpy as np
import cv2
import sortContours as sort

# constants
MIN_CONTOUR_AREA = 10
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30



# ====================
# main function; starts training process
# ====================
def classifyImage(trainingFileName, classificationArray):
	classificationFile = file('classification_labels.txt', 'a')
	trainingDataFile = file('training_data.txt', 'a')
	
	# read in training numbers image, grayscale it, filter to black and white, display image, make a copy
	trainingImg = cv2.imread(trainingFileName)
	grayImg = cv2.cvtColor(trainingImg, cv2.COLOR_BGR2GRAY)
	threshImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	cv2.imshow("threshImg", threshImg)
	threshImgCopy = threshImg.copy()

	# find and sort contours
	contours, hierarchy = cv2.findContours(threshImgCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key=lambda x: sort.sortContours(x))
	

	# declare empty numpy array for each training data sample
	trainingdata = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

	# show found contours using bounding rect
	for contour in contours:
			if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
					[intX, intY, intW, intH] = cv2.boundingRect(contour)
					cv2.rectangle(trainingImg,(intX, intY),(intX + intW, intY + intH),(255, 0, 255),1)

					contourImg = threshImg[intY:intY + intH, intX:intX + intW]
					contourImgResized = cv2.resize(contourImg, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
					
					cv2.imshow("training_numbers",trainingImg)
					sampleLetter = contourImgResized.reshape((1,RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
					trainingdata = np.append(trainingdata, sampleLetter,0)  # add current flattened impage numpy array to list of flattened image numpy arrays
	
	cv2.waitKey(0)
	
	# convert classifications list of ints to numpy array of floats
	floatClassifications = np.array(classificationArray, np.float32)

	# flatten to 1d
	flatClassifications = floatClassifications.reshape((floatClassifications.size, 1))

	print "\n\ntraining complete !!\n"


	np.savetxt(classificationFile, flatClassifications)
	classificationFile.close()
	
	np.savetxt(trainingDataFile, trainingdata)
	trainingDataFile.close()

	cv2.destroyAllWindows()  # remove windows from memory

	return


# ====================
# Run main methods
# ====================
couriernew_lowercase_labels = [ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
	ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
	ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]
couriernew_uppercase_labels = [ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
	ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
	ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
couriernew_numbers_labels = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

classifyImage("traindata/couriernew_lowercase.png", couriernew_lowercase_labels)
classifyImage("traindata/couriernew_uppercase.png", couriernew_uppercase_labels)
classifyImage("traindata/couriernew_numbers.png", couriernew_numbers_labels)

