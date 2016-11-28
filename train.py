import os
import numpy as np
import cv2
import ContourHelper as help

# Constants
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

CLASSIFICATION_FILENAME = 'classification_labels.txt'
TRAINING_DATA_FILENAME = 'training_data.txt'
TRAIN_DATA_DIR = ".\\traindata"

# Flags
showImages = False # whether to cv2.imshow() the results
showContourOrder = True # whether to show order of contours being classified
checkForTittles = False

# Classification Labels
lowercase_labels = [ord('a'), ord('b'), ord('c'), ord('d'), ord('e'), ord('f'), ord('g'), ord('h'), ord('i'), ord('j'),
	ord('k'), ord('l'), ord('m'), ord('n'), ord('o'), ord('p'), ord('q'), ord('r'), ord('s'), ord('t'),
	ord('u'), ord('v'), ord('w'), ord('x'), ord('y'), ord('z')]
uppercase_labels = [ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
	ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
	ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
numbers_labels = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

def classifyImage(trainingImageName, classificationArray):
	"""
	Produce training data and classification files given filename and classification array
	:param trainingImageName:
	:param classificationArray:
	:return:
	"""
	# open or create classification and training data files
	classificationFile = file(CLASSIFICATION_FILENAME, 'a')
	trainingDataFile = file(TRAINING_DATA_FILENAME, 'a')
	
	# read in training image and apply preprocessing functions
	trainingImg = cv2.imread(trainingImageName)
	grayImg = cv2.cvtColor(trainingImg, cv2.COLOR_BGR2GRAY)
	blurImg = cv2.blur(grayImg,(5,5))
	threshImg = cv2.adaptiveThreshold(blurImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

	# find and sort contours
	contours, hierarchy = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key=lambda x: help.sortContoursUpperLeftToLowerRight(x))
	
	# declare empty array with size equal to number of training data samples
	trainingdata = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
	
	# find contours that have a tittle on top (i's or j's)
	if checkForTittles:
		lettersWithTittles = []
		tittles = []
		for i in range(len(contours)-1):
			if help.detectTittles(contours[i], contours[i+1]):
				lettersWithTittles.append(contours[i])
				tittles.append(contours[i+1])

	# add appropriate contours to training data
	for contour in contours:
			if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
				# get bounding rect for current contour
				[intX, intY, intW, intH] = cv2.boundingRect(contour)

				if checkForTittles and help.getIndexOfTittle(contour, lettersWithTittles) > -1:
					index = help.getIndexOfTittle(contour, lettersWithTittles)
					if index > -1:
						# get dimensions of tittle and use to draw rect and grab letter
						[tX, tY,tWidth, tHeight] = cv2.boundingRect(tittles[index])
						additionalHeight = intY - (tY + tHeight)

						cv2.rectangle(trainingImg,(intX, tY),(intX + intW, tY + intH + tHeight + additionalHeight),(255, 0, 0),1)
						contourImg = threshImg[intY:intY + intH + tHeight + additionalHeight, intX:intX + intW]
				else:
					# draw rect and grab letter
					cv2.rectangle(trainingImg, (intX, intY), (intX + intW, intY + intH), (255, 0, 255), 1)
					contourImg = threshImg[intY:intY + intH, intX:intX + intW]

				# resize image and show on original
				contourImgResized = cv2.resize(contourImg, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
				if showImages:
					cv2.imshow(trainingImageName + " thresh		olded", threshImg)
					cv2.imshow(trainingImageName, trainingImg)
				
				# flatten contour to 1D and append to training data
				contourImgFlatten = contourImgResized.reshape((1,RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
				trainingdata = np.append(trainingdata, contourImgFlatten,0)

				# show order that contours are classified
				if showImages and showContourOrder:
					cv2.imshow(trainingImageName, trainingImg)
					cv2.waitKey(0)
		
	# convert classifications list of ints to numpy array of floats
	floatClassifications = np.array(classificationArray, np.float32)

	# flatten to 1d
	flatClassifications = floatClassifications.reshape((floatClassifications.size, 1))

	# save classification and training data
	np.savetxt(classificationFile, flatClassifications)
	classificationFile.close()
	
	np.savetxt(trainingDataFile, trainingdata)
	trainingDataFile.close()

	# remove windows from memory
	cv2.waitKey(0)
	cv2.destroyAllWindows() 
	print "Training complete: %s" % trainingImageName

	return


def main():
	"""
	Classifies training data images with uppercase, lowercase, and number labels
	:return: void
	"""

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~
	# TODO Call classifyImage based on filename endings in the train data folder
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~	 
	open(CLASSIFICATION_FILENAME, 'w').close()
	open(TRAINING_DATA_FILENAME, 'w').close()
	
	for file in os.listdir(TRAIN_DATA_DIR):
		label = []
		#if file.endswith("_lowercase.png"):
			#	label.append(lowercase_labels)
		if file.endswith("_uppercase.png"):
			label.append(uppercase_labels)
		elif file.endswith("_numbers.png"):
			label.append(numbers_labels)
		else:
			continue
			
		filePath = os.path.join(TRAIN_DATA_DIR, file)
		classifyImage(filePath, label)
	"""
	classifyImage("traindata/handwriting/kyla_uppercase.jpg", uppercase_labels)
	classifyImage("traindata/handwriting/kyla_numbers.jpg", numbers_labels)
	classifyImage("traindata/arial_uppercase.png", uppercase_labels)
	classifyImage("traindata/arial_numbers.png", numbers_labels)
	classifyImage("traindata/calibri_uppercase.png", uppercase_labels)
	classifyImage("traindata/calibri_numbers.png", numbers_labels)
	classifyImage("traindata/chalkboard_uppercase.png", uppercase_labels)
	classifyImage("traindata/chalkboard_numbers.png", numbers_labels)
	classifyImage("traindata/comicsans_uppercase.png", uppercase_labels)
	classifyImage("traindata/comicsans_numbers.png", numbers_labels)
	classifyImage("traindata/couriernew_uppercase.png", uppercase_labels)
	classifyImage("traindata/couriernew_numbers.png", numbers_labels)
	classifyImage("traindata/helvetica_uppercase.png", uppercase_labels)
	classifyImage("traindata/helvetica_numbers.png", numbers_labels)
	classifyImage("traindata/TNR_uppercase.png", uppercase_labels)
	classifyImage("traindata/TNR_numbers.png", numbers_labels)
	classifyImage("traindata/verdana_uppercase.png", uppercase_labels)
	classifyImage("traindata/verdana_numbers.png", numbers_labels)
	"""
	return

if __name__ == "__main__":
	main()
