# Main Contributor: Kyla Bouldin
# Other Contributors: Yash, Edrienne
# Description: Tests the trained data on test images
# TODO: Method docs and refactoring

import math
import cv2
import numpy as np
import operator
import os

import Trainer
import CharacterHelper

# Constants
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class ContourWithData():
    """
    Class to help manipulate contours.
    """
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        """
        Calculate bounding rectangle information
        :return:
        """
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        """
        Checks if the contour matches criteria
        :return:
        """
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True

class Character():
    """
    Represents a single character
    """

    def __init__(self, _contour):
        self.contour = _contour

        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)


def __readImage(imgFilename):
    """
    Returns image with specified fileName
    :param imgFilename:
    :return: imgTest
    """
    imgTest = cv2.imread(imgFilename)  # read in testing numbers image
    if imgTest is None:  # if image was not read successfully
        print "error: image not read from file \n\n"  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return  # and exit function (which exits program)

    return imgTest

def __preprocessImage(img):
    """
    Converts the image to gray scale and then thresholds the image
    :param img:
    :return: imgThresh
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)  # gaussian blur

    # Grayscale image to BW image
    imgThresh = cv2.adaptiveThreshold(imgBlurred,  # input image
                                      255,  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Gaussian threshold
                                      cv2.THRESH_BINARY_INV, # invert so foreground white, background black
                                      11,  # size of a pixel neighborhood used to calculate threshold value
                                      2)  # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()  # make a copy of the thresh image, this in necessary b/c findContours modifies the image

    return imgThresh

def __getValidContoursWithData(imgThresh):
    """
    Gets valid contours with data from the thresholded image
    :param imgThresh:
    :return: validContoursWithData
    """
    allContoursWithData = []  # declare empty lists,
    validContoursWithData = []  # we will fill these shortly
    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,
                                                 # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,  # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)  # compress horizontal, vertical, and diagonal segments and leave only their end points
    npaContours.sort(key=lambda x:Trainer.__sortContours(x))


    # Create ContourWithData objects for each contour
    for npaContour in npaContours:  # for each contour
        contourWithData = ContourWithData()  # instantiate a contour with data object
        contourWithData.npaContour = npaContour  # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)  # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()  # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)  # calculate the contour area
        allContoursWithData.append(contourWithData)  # add contour with data object to list of all contours with data

    # Determine valid ContourWithData objects
    for contourWithData in allContoursWithData:  # for all contours
        if contourWithData.checkIfContourIsValid():  # check if valid
            validContoursWithData.append(contourWithData)  # if so, append to valid contour list

    return validContoursWithData

def __getCharactersFromContoursWithData(imgTest, imgThresh, validContoursWithData, kNearest):
    """
    Returns String of characters that correspond to the contours
    :param imgTest:
    :param imgThresh:
    :param validContoursWithData:
    :param kNearest:
    :return: strFinalString
    """
    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
        # Draw a green rect around the current char
        cv2.rectangle(imgTest,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        # crop char out of threshold image
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        # resize image, this will be more consistent for recognition and storage
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # flatten image into 1d numpy array
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        # convert from 1d numpy array of ints to 1d numpy array of floats
        npaROIResized = np.float32(npaROIResized)
        # call KNN function find_nearest
        retval, npaResults, neigh_resp, dists = kNearest.find_nearest(npaROIResized, k = 1)
        # get character from results
        strCurrentChar = str(chr(int(npaResults[0][0])))
        # append current char to full string
        strFinalString = strFinalString + strCurrentChar

    return strFinalString

def __getStringsFromContoursWithData(imgTest, imgThresh, validContoursWithData, kNearest):
    """
    Returns String of characters that correspond to the contours
    :param imgTest:
    :param imgThresh:
    :param validContoursWithData:
    :param kNearest:
    :return: strFinalString
    """
    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    characterList = []
    for contourWithData in validContoursWithData:
        # Draw a green rect around the current char
        character = Character(contourWithData.npaContour)
        characterList.append(character)

    # Get mean distance between characters
    nextChar = ""
    distanceList = []
    for i in range(len(characterList) - 1):
        character = characterList[i]
        nextChar = characterList[i + 1]
        distance = CharacterHelper.getDistanceBetween(nextChar, character)
        distanceList.append(distance)

    meanDistance = np.mean(distanceList)
    stdDev = np.std(distanceList)

    nextChar = ""
    for i in range(len(characterList)):
        character = characterList[i]

        imgROI = imgThresh[character.intBoundingRectY: character.intBoundingRectY + character.intBoundingRectHeight,
                     character.intBoundingRectX: character.intBoundingRectX + character.intBoundingRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.find_nearest(npaROIResized, k=1)

        # Get character
        strCurrentChar = str(chr(int(npaResults[0][0])))
        strFinalString = strFinalString + strCurrentChar

        # Detect space
        if ( i < len(characterList) - 1):
            nextChar = characterList[i + 1]
            distance = CharacterHelper.getDistanceBetween(nextChar, character)
            if (abs(distance - meanDistance) > stdDev):
                strFinalString = strFinalString + " "


    return strFinalString

def main():
    # Trainer.trainData("training_letters.png")
    kNearest = Trainer.getTrainedKNN()
    imgTest = __readImage("../handwrittendata/real4.jpg")
    # imgTest = __readImage("testimage2.png");
    imgThresh = __preprocessImage(imgTest)
    validContoursWithData = __getValidContoursWithData(imgThresh)
    strFinalString = __getCharactersFromContoursWithData(imgTest, imgThresh, validContoursWithData, kNearest)
    strFinalString = __getStringsFromContoursWithData(imgTest, imgThresh, validContoursWithData, kNearest)

    print "\n" + strFinalString + "\n" # string of characters in image
    cv2.imshow("imgTest", imgTest) # image with green boxes drawn around characters
    cv2.waitKey(0)
    cv2.destroyAllWindows()             # remove windows from memory
    return

if __name__ == "__main__":
    main()
