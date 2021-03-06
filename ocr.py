# ocr.py
# Runs OCR algorithm
# Contributors: Kyla, Edrienne

import cv2
import numpy as np
import contour_helper as help

import train

# Constants
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
K = 2 # for k-NN

# Character Contour Criteria
MIN_WIDTH = 2
MIN_HEIGHT = 5
MIN_CONTOUR_AREA = 100
MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 3.9

# Flags
showImages = True
showContourOrder = True

def __getTrainedKNearest():
	"""
	Runs K-Nearest Neighbors classification using classification and training data
	:return: kNearest object
	"""
	try:
		classificationLabels = np.loadtxt(train.CLASSIFICATION_FILENAME, np.float32)
	except:
		print("Can't find classification labels")
	
	try:
		trainingData = np.loadtxt(train.TRAINING_DATA_FILENAME, np.float32)
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
	# TODO: Look into more preprocessing algorithms.
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgBlur = cv2.blur(imgGray, (5, 5))
	imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return imgThresh


# TODO: Move all generalizable, independent contour manipulating methods belong in ContourHelper
def __filterForCharacterContours(allContours):
    """
    Eliminates contours that don't match character criteria
    :param allContours:
    """
    characterContourList = []

    # TODO: Evaluate whether it's worth checking contour area at all
    # get average contour area and mean box area
    contourAreaList, meanBoxAreaList = [], []
    for contour in allContours:
        contourAreaList.append(cv2.contourArea(contour))
        [x, y, width, height] = cv2.boundingRect(contour)
        meanBoxAreaList.append(width * height)

    meanContourArea, stdDevContourArea, meanBoxArea, stdDevBoxArea = 0, 0, 0, 0

    if (len(contourAreaList) > 0):
        meanContourArea = np.mean(contourAreaList)
        stdDevContourArea = np.std(contourAreaList)
    if (len(meanBoxAreaList) > 0):
        meanBoxArea = np.mean(meanBoxAreaList)
        stdDevBoxArea = np.std(meanBoxAreaList)

    # filter by aspect ratio, area
    for contour in allContours:
        [intX, intY, intWidth, intHeight] = cv2.boundingRect(contour)
        aspectRatio = float(intWidth) / float(intHeight)

        if cv2.contourArea(contour) > MIN_CONTOUR_AREA \
                and abs(intWidth * intHeight - meanBoxArea) <= 3 * stdDevBoxArea \
				and MIN_ASPECT_RATIO < aspectRatio < MAX_ASPECT_RATIO \
                and True:
			characterContourList.append(contour)

    # TODO: Research more geometric filters we can apply and other methods people use
    characterContourList.sort(key=lambda c: help.sortContoursUpperLeftToLowerRight(c))

    return characterContourList


# TODO: Debug! Not working for all cases.
def __removeOverlappingContours(contourList):
	"""
	If two contours overlap, remove the smaller, inner one
	:param contourList:
	:return: contourListWithRemoval
	"""
	contourListWithRemoval = list(contourList)

	for i in range(len(contourList)):
		for j in range(len(contourList)):
			if i == j:
				continue # skip comparison to self

			contourA = contourList[i]
			contourB = contourList[j]
			[intX_A, intY_A, intWidth_A, intHeight_A] = cv2.boundingRect(contourA)
			[intX_B, intY_B, intWidth_B, intHeight_B] = cv2.boundingRect(contourB)

			# If bounding boxes collide, remove the smaller one
			if intY_A + intHeight_A <= intX_B or \
				intY_A >= intY_B + intHeight_B or \
				intX_A + intWidth_A <= intX_B or \
				intX_A >= intX_B + intY_B:
				rectArea_A = intWidth_A * intHeight_A
				rectArea_B = intWidth_B * intHeight_B

				if rectArea_A < rectArea_B and contourA in contourListWithRemoval:
					contourListWithRemoval.remove(contourA)
				elif contourB in contourListWithRemoval:
					contourListWithRemoval.remove(contourB)

	return contourListWithRemoval


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
		ret, result, neighbors, dist = kNearest.find_nearest(letter, K)

		# append character to string
		currentChar = str(chr(int(ret)))
		print currentChar
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


def recognizeCharacters(fileName):
	"""
	Runs optical character recognition (OCR) algorithm to detect and print characters in an image.
	:param fileName: image file path
	:return text
	"""

	# TODO: Create comprehensive preprocess class with various preprocessing functions
	testImage = cv2.imread(fileName)
	imgThresh = __preprocessImage(testImage)

	# find character contours for characters and sort from upper left to lower right
	allContours, hierarchy = cv2.findContours(imgThresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	allContours.sort(key=lambda x: help.sortContoursUpperLeftToLowerRight(x))
	characterContourList = __filterForCharacterContours(allContours)

	kNearest = __getTrainedKNearest()
	text = __getStringFromCharacterContours(testImage, imgThresh, characterContourList, kNearest)

	if showImages:
		cv2.imshow("imgThresh", imgThresh)
		cv2.imshow("testImage", testImage)
		cv2.waitKey(0)

	cv2.destroyAllWindows()
			
	return text


def main():
	# Hard-coded testing
	print recognizeCharacters("testdata/handwritten/real3.jpg")
	"""
	print recognizeCharacters("testdata/couriernew_all.png")
	print recognizeCharacters("testdata/couriernew_helloworld_upper.png")
	print recognizeCharacters("testdata/foobar.png")
	print recognizeCharacters("testdata/licenseplate_upper+digits.png")
	print recognizeCharacters("testdata/multiline_number.png")
	print recognizeCharacters("testdata/multiline.png")
	print recognizeCharacters("testdata/timesnewroman_digits.png")

	print recognizeCharacters("testdata/handwritten/kyla_letters.jpg")
	print recognizeCharacters("testdata/handwritten/kyla_mix.jpg")
	print recognizeCharacters("testdata/handwritten/kyla_numbers.jpg")
	print recognizeCharacters("testdata/handwritten/licenseplate.jpg")

	print recognizeCharacters("testdata/handwritten/real1.jpg")
	print recognizeCharacters("testdata/handwritten/real2.jpg")
	print recognizeCharacters("testdata/handwritten/real3.jpg")
	print recognizeCharacters("testdata/handwritten/real4.jpg")
	"""
if __name__ == "__main__":
	main()