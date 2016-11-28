import cv2
import numpy as np
import ContourHelper as help

import Train

# Constants
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Character Contour Criteria
MIN_WIDTH = 1
MIN_HEIGHT = 5

MIN_CONTOUR_AREA = 100
MIN_BOX_AREA_DIFF = 1700

MIN_ASPECT_RATIO = 0.14
MAX_ASPECT_RATIO = 4.0

# Flags
showImages = True
showContourOrder = True

def __getTrainedKNearest():
	"""
	Runs K-Nearest Neighbors classification using classification and training data
	:return: kNearest object
	"""
	try:
		classificationLabels = np.loadtxt(Train.CLASSIFICATION_FILENAME, np.float32)
	except:
		print("Can't find classification labels")
	
	try:
		trainingData = np.loadtxt(Train.TRAINING_DATA_FILENAME, np.float32)
	except:
		print("Can't find training data")

	classificationLabels = classificationLabels.reshape((classificationLabels.size, 1))  # reshape to 1D for train()

	# create KNearest and train
	kNearest = cv2.KNearest()
	kNearest.train(trainingData, classificationLabels)

	return kNearest


def __preprocessImage(img):
	"""
	Processes image to make it easier to find contours we want
	Converts to grayscale, blurs, and binarizes using thresholding
	:param img:
	:return: imgThresh Binarized image
	"""
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.blur(imgGray, (5, 5))
	imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return imgThresh


def __filterForCharacterContours(allContours):
    """
	Eliminates contours that don't match character criteria
	:param allContours:
	:return: characterContourList
	"""
    characterContourList = []

    # TODO: Should we do contour area or box area?
    # get average contour area
    areaList = []
    for i in allContours:
    	areaList.append(cv2.contourArea(i))
    meanArea, stdDevArea = 0, 0
    if (len(areaList) > 0):
        meanArea = np.mean(areaList)
        stdDevArea = np.std(areaList)

    # get mean box area
    meanBoxAreaList = []
    for contour in characterContourList:
        [x, y, width, height] = cv2.boundingRect(contour)
        meanBoxAreaList.append(width * height)
    meanBoxArea = 0
    if (len(meanBoxAreaList) > 0):
        meanBoxArea = np.mean(meanBoxAreaList)

    # filter by aspect ratio, area
    for contour in allContours:
        [intX, intY, intWidth, intHeight] = cv2.boundingRect(contour)
        aspectRatio = float(intWidth) / float(intHeight)

        if cv2.contourArea(contour) > MIN_CONTOUR_AREA \
                and abs(cv2.contourArea(i) - meanArea) <= 2 * stdDevArea \
                and MIN_ASPECT_RATIO < aspectRatio < MAX_ASPECT_RATIO:
            characterContourList.append(contour)

    # TODO: add more filters
    characterContourList.sort(key=lambda c: help.sortContoursUpperLeftToLowerRight(c))

    return characterContourList


def __getStringFromCharacterContours(testImage, imgThresh, characterContourList, kNearest):
	"""
	Returns a string containing the characters detected from contours using kNN. Segments words by spaces
	:param testImage: Original test image
	:param imgThresh: Thresholded test image
	:param characterContourList: List of contours
	:param kNearest: KNearest object that has already been trained
	:return: finalString
	"""
	meanDistance, stdDev = help.getMeanDistanceBetweenContours(characterContourList)
	finalString = ""

	for i in range(len(characterContourList)):
		charContour = characterContourList[i]
		[intX, intY, intWidth, intHeight] = cv2.boundingRect(charContour)
		cv2.rectangle(testImage, (intX, intY), (intX + intWidth, intY + intHeight), (0, 255, 0), 2)

		# crop and resize image
		letter = imgThresh[intY: intY + intHeight, intX: intX + intWidth]
		letter = cv2.resize(letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

		# flatten and convert to numpy array of floats
		letter = letter.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
		letter = np.float32(letter)

		# find k nearest neighbor to determine character
		ret, result, neighbors, dist = kNearest.find_nearest(letter, k=3)

		# append character to string
		currentChar = str(chr(int(ret)))
		finalString = finalString + currentChar

		# detect space
		if (i < len(characterContourList) - 1):
			nextChar = characterContourList[i + 1]
			distance = help.getDistanceBetween(nextChar, charContour)
			if (abs(distance - meanDistance) > stdDev):
				finalString = finalString + " "

		# show contour order
		if showImages and showContourOrder:
			cv2.imshow("testImage", testImage)
			cv2.waitKey(0)

	return finalString


def printImageCharacters(fileName):
	"""
	Runs optical character recognition (OCR) algorithm to detect and print characters in an image.
	:param fileName: image file path
	"""

	# TODO: Create comprehensive preprocess class with various preprocessing functions
	testImage = cv2.imread(fileName)
	imgThresh = __preprocessImage(testImage)

	if showImages:
		cv2.imshow("imgThresh", imgThresh)
		cv2.waitKey(0)

	# find character contours for characters and sort from upper left to lower right
	allContours, hierarchy = cv2.findContours(imgThresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	characterContourList = __filterForCharacterContours(allContours)

	kNearest = __getTrainedKNearest()
	text = __getStringFromCharacterContours(testImage, imgThresh, characterContourList, kNearest)

	if showImages:
		cv2.imshow("testImage", testImage)
		cv2.waitKey(0)

	cv2.destroyAllWindows()
			
	print text


def main():
	# TODO: Rename this class something else, like Main or OCR
	printImageCharacters("testdata/couriernew_test.png")
	printImageCharacters("testdata/couriernew_helloworld.png")
	printImageCharacters("testdata/tnr_helloworld.png")
	printImageCharacters("handwrittendata/real2.jpg")


if __name__ == "__main__":
	main()