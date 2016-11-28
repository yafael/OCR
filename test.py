import cv2
import numpy as np
import helperfunctions as help

MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# ====================
# finds contours on given image file and prints detected string
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
	
	# reshape classifications to 1d
	classificationLabels = classificationLabels.reshape((classificationLabels.size, 1))
	
	# create KNearest and train
	kNearest = cv2.KNearest()
	kNearest.train(trainingData, classificationLabels)
		
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~
	# TODO UPDATE PREPROCESSING FUNCTIONS AND OR PLACE IN HELPER FUNCTIONS
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~	 
		
	# read in testing image and apply preprocessing functions
	testImage = cv2.imread(testFileName)
	testImGray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
	imgThresh = cv2.adaptiveThreshold(testImGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	imgThreshCopy = imgThresh.copy() 
	
	# find and sort contours
	contours, heierachy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours.sort(key=lambda x: help.sortContours(x))

	# initialize variables
	finalString = ""
	characterContourList = []
	
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~
	# TODO GET MEAN CONTOUR AREA AND ADD THRESHOLD BETWEEN MEAN AND CURRENT TO FILTER CONTOURS
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~	 
	for contour in contours:
		area = cv2.contourArea(contour) 
		
		if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
			characterContourList.append(contour)
	
	# get mean distance between contours
	nextChar = ""
	distanceList = []
	for i in range(len(characterContourList) - 1):
		character = characterContourList[i]
		nextChar = characterContourList[i + 1]
		distance = help.getDistanceBetween(nextChar, character)
		distanceList.append(distance)
	
	meanDistance = np.mean(distanceList)
	stdDev = np.std(distanceList)
	
	# get get bounding rects from selected contours and detect spaces
	for i in range(len(characterContourList)):
		[intX, intY, intWidth, intHeight] = cv2.boundingRect(characterContourList[i])
		cv2.rectangle(testImage,(intX, intY),(intX + intWidth, intY + intHeight), (0, 255, 0), 2)

		# crop and resize image
		letter = imgThresh[intY : intY + intHeight, intX : intX + intWidth]
		letter = cv2.resize(letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
		
		# flatten and convert to numpy array of floats
		letter = letter.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
		letter = np.float32(letter)
		
		# find k nearest neighbor
		ret, result, neighbors, dist = kNearest.find_nearest(letter, k=3)
		
		# append to string
		currentChar = str(chr(int(ret)))
		finalString = finalString + currentChar
		
		cv2.imshow("testImage", testImage)
		cv2.waitKey(0)

		#detect space
		nextChar = ""
		if (i < len(characterContourList) - 1):
			nextChar = characterContourList[i + 1]
			distance = help.getDistanceBetween(nextChar, character)
			if (abs(distance - meanDistance) > stdDev):
				finalString = finalString + " "
			
	print "\n" + finalString + "\n"

	cv2.destroyAllWindows()
			
	return finalString

# ====================
# run findContours on multiple images
# ====================
findContours("testdata/couriernew_test.png")
findContours("testdata/couriernew_helloworld.png")
findContours("testdata/tnr_helloworld.png")
findContours("handwrittendata/real2.jpg")