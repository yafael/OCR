# Main Contributor: Kyla Bouldin
# Other Contributors: Yashvardhan Gusani, Edrienne Co
# Description: Creates training data file
# TODO: Fill out method information.

import sys, os
import numpy as np
import cv2
import argparse

# Constants
MIN_CONTOUR_AREA = 0
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Declare characters we are interested in (A-Z, 0-9)
intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

def __sortContours(point1):
    """
    Sorts contours from left to right, top to bottom
    :param point1:
    :return:
    """
    tolerance_factor = 50
    mom1 = cv2.moments(point1)

    x, y = 0, 0
    if mom1["m00"] != 0:
        x = int(mom1["m10"] / mom1["m00"])
        y = int(mom1["m01"] / mom1["m00"])

    return((y // tolerance_factor) * tolerance_factor) * 100 + x

def __preprocessImage(img):
    """
    Converts the image to gray scale and then thresholds the image
    :param img:
    :return: imgThresh
    """

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)       
    return imgThresh 

def __getContours(img):
    """
    Gets contours, flattened images, thresholded image from a training image
    :param img:
    :return:
    """
    imgThresh = __preprocessImage(img)
    imgThreshCopy = imgThresh.copy()
    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    npaContours.sort(key=lambda x:__sortContours(x))
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))   

    return npaContours, npaFlattenedImages, imgThresh

def __processContours(img, npaContours, npaFlattenedImages, imgThresh):
    """
    Produces classification and flattened image data from contours in an image
    :param img
    :param npaContours:
    :param npaFlattenedImages:
    :param imgThresh:
    :return: intClassifications, npaFlattenedImages
    """
    intClassifications = []

    for npaContour in npaContours:  # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:  # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)  # get and break out bounding rect

            # Draw rectangle around each contour as we ask user for input
            cv2.rectangle(img,  # draw rectangle on original training image
                          (intX, intY),  # upper left corner
                          (intX + intW, intY + intH),  # lower right corner
                          (0, 0, 255),  # red
                          2)  # thickness

            # Crop char out of threshold image and resize to be consistent with recognition and storage
            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("Training Image", img)  # training numbers image with drawn rectangles
            intChar = cv2.waitKey(0)  # get key press
            if intChar == 27:
                sys.exit()  # exit when esc key pressed
            elif intChar in intValidChars:
                intClassifications.append(intChar)
                flattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, flattenedImage, 0)
                print intClassifications
                print npaFlattenedImages

    return npaFlattenedImages, intClassifications

def trainData(imageFilename):
    """
    Generates classification and flattened image information for a particular image file
    :param imageFilename:
    :return:
    """
    imgTraining = cv2.imread(imageFilename)
    npaContours, npaFlattenedImages, imgThresh = __getContours(imgTraining)
    npaFlattenedImages, intClassifications = __processContours(imgTraining, npaContours, npaFlattenedImages, imgThresh)

    fltClassifications = np.array(intClassifications, np.float32) # convert from ints to floats
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1)) # flatten to 1D array

    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    print "\n\nTraining complete.\n"
    cv2.destroyAllWindows()

    return

def getTrainedKNN():
    """
    Returns KNN trained on trained classification and flattened image data
    :return:
    """
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        os.system("pause")
        return

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        os.system("pause")
        return

    # Instantiate KNN and train against flattened images and classifications.
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))  # reshape to 1D for train()

    kNearest = cv2.KNearest()
    kNearest.train(npaFlattenedImages, npaClassifications)

    return kNearest

def main():
    """
    Main function for training data.
    Call when testing.
    :return:
    """
    trainData("training_letters.png")

if __name__ == "__main__":
    main()