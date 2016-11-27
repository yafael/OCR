# Main Contributor: Edrienne
# Description: Finds regions of text in an image

import math

import cv2
import numpy as np

import CharacterHelper
import ImageHelper

# Constants
TEXT_REGION_WIDTH_PADDING_FACTOR = 1.3
TEXT_REGION_HEIGHT_PADDING_FACTOR = 1.5


class CandidateCharacter():
    """
    Represents a region in an image that (possibly) contains a single character
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


class CandidateTextRegion():
    """
    Represents a region in an image that (possibly) contains text, AKA a string of characters
    """

    def __init__(self):
        self.imgText = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfTextInScene = None

        self.strChars = ""


def getCandidateTextRegions(img):
    """
    Returns candidate text regions from the image
    :param img:
    :return: arrCandidateTextRegions
    """
    arrCandidateTextRegions = []

    height, width, numChannels = img.shape

    imgGrayscale = np.zeros((height, width, 1), np.uint8)
    imgThresh = np.zeros((height, width, 1), np.uint8)
    imgContours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()
    imgGrayscale, imgThresh = ImageHelper.preprocessNaturalImage(img)

    listOfCandidateCharactersInImg = __findCandidateCharactersInImg(imgThresh)
    listOfListsOfMatchingCharsInImg = CharacterHelper.findListOfListsOfMatchingChars(listOfCandidateCharactersInImg)

    for listOfMatchingChars in listOfListsOfMatchingCharsInImg:
        # Extract text region for each group of matching characters
        candidateTextRegion = __extractTextRegion(img, listOfMatchingChars)

        if candidateTextRegion.imgText is not None:
            arrCandidateTextRegions.append(candidateTextRegion)

    print "\n" + str(len(arrCandidateTextRegions)) + " text regions found"

    return arrCandidateTextRegions

#########################
## PRIVATE METHODS
#########################
def __findCandidateCharactersInImg(imgThresh):
    """
    Find all contours in a thresholded image and return CandidateCharacter objects that correspond to valid contours
    :param imgThresh:
    :return: arrCandidateCharacters
    """
    arrCandidateCharacters = []
    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        candidateCharacter = CandidateCharacter(contours[i])

        if CharacterHelper.checkIfCandidateCharacter(candidateCharacter):                   #
            arrCandidateCharacters.append(candidateCharacter)

    return arrCandidateCharacters


def __extractTextRegion(imgOriginal, listOfMatchingChars):
    """
    Extracts a text region of the image that has the list of matching characters.
    Gets proper height, width, and orientation before cropping.
    :param imgOriginal:
    :param listOfMatchingChars:
    :return: candidateTextRegion
    """
    candidateTextRegion = CandidateTextRegion()
    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX) # sort L to R

    # Calculate center of text region
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # Calculate text region width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * TEXT_REGION_WIDTH_PADDING_FACTOR)
    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * TEXT_REGION_HEIGHT_PADDING_FACTOR)

    # Calculate correction angle of text region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = CharacterHelper.getDistanceBetween(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # Pack plate region center point, width, height, and correction angle into rotated rect member variable of plate
    candidateTextRegion.rrLocationOfTextInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

    # Get rotation matrix for calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width and height
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    candidateTextRegion.imgText = imgCropped

    return candidateTextRegion
