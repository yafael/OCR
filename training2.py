# Author: Kyla Bouldin, Yashvardhan Gusani
# Description: Creates training data file

import sys
import numpy as np
import cv2
import argparse

# Constants
MIN_CONTOUR_AREA = 0

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# ====================
# Custom function designed to sort the contours from left to right, top to bottom
# ====================
def sortContours(point1):
    tolerance_factor = 50
    mom1 = cv2.moments(point1)

    x = int(mom1['m10']/mom1['m00'])
    y = int(mom1['m01']/mom1['m00'])  

    return((y // tolerance_factor) * tolerance_factor) * 100 + x

# ====================
# Main function
# ====================
def main():
	# Read in training numbers image
    imgTrainingNumbers = cv2.imread("training_letters.png")
	# Get grayscale image
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    # Filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Show threshold image for reference
    cv2.imshow("imgThresh", imgThresh)
	# Copy thresh image for findCountours to modify
    imgThreshCopy = imgThresh.copy()

	# Find outermost contours
    # Compress horizontal, vertical, and diagonal segments and leave only their end points
    image, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #################################
    # Sorting Contours in the correct manner
    #################################
    npaContours.sort(key=lambda x:sortContours(x))

	# Declare empty numpy array for writing to file later
	# Zero rows, enough cols to hold all image data
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # Declare empty classifications list
    # Will use to classify chars from user input
    # Will write to file at the end
    intClassifications = []

		
    #################################
    # TODO: UPDATE THIS ONCE SORTING WORKS
    #################################
	# Declare characters we are interested in (A-Z, 0-9)
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

            # Draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            # Crop char out of threshold image and resize to be consistent with recognition and storage
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI) # cropped out char
            cv2.imshow("imgROIResized", imgROIResized) # resized image
            cv2.imshow("training_numbers.png", imgTrainingNumbers) # training numbers image with drawn rectangles

            intChar = cv2.waitKey(0)            # get key press

            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .
                # Append classification char to integer list of chars
                # Will convert to float later before writing to file
                intClassifications.append(intChar)

                # Flatten image to 1d numpy array so we can write to file later
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                # Add current flattened impage numpy array to list of flattened image numpy arrays
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
            # end if
        # end if
    # end for

    # Convert classifications list of ints to numpy array of floats
    fltClassifications = np.array(intClassifications, np.float32)
    # Flatten numpy array of floats to 1d so we can write to file later
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print "\n\nTraining complete !!\n"

    # Write flattened images and classifications to file
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    cv2.destroyAllWindows() # remove windows from memory

    return

main()



