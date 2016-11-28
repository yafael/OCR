# Main Contributor: Edrienne
# Description: Contains different functions dealing with Characters in an image
# TODO: Derived from tutorial, give credit

import math

import OCR
import cv2
import numpy as np

import ImageHelper
import TextRegionFinder
import Trainer

kNearest = cv2.KNearest()

# Constants
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100


def checkIfCandidateCharacter(candidateCharacter):
    if (candidateCharacter.intBoundingRectArea > MIN_PIXEL_AREA and
                candidateCharacter.intBoundingRectWidth > MIN_PIXEL_WIDTH and
                candidateCharacter.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
                    MIN_ASPECT_RATIO < candidateCharacter.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False

def findListOfListsOfMatchingChars(listOfCandidateCharacters):
    """
    Given list of all possible characters, rearrange the list into a list of lists of matching characters.
    :param listOfCandidateCharacters:
    :return: listOfListsOfMatchingChars
    """
    listOfListsOfMatchingChars = []

    for candidateCharacter in listOfCandidateCharacters:
        # Find all chars in the big list that match the current char
        listOfMatchingChars = __findListOfMatchingChars(candidateCharacter, listOfCandidateCharacters)

        # Add the current char to current possible list of matching chars
        listOfMatchingChars.append(candidateCharacter)

        if len(listOfMatchingChars) >= MIN_NUMBER_OF_MATCHING_CHARS:
            listOfListsOfMatchingChars.append(listOfMatchingChars)
            listOfCandidateCharactersWithCurrentMatchesRemoved = []

            # Remove matches from list of possible characters to avoid duplicates
            listOfCandidateCharactersWithCurrentMatchesRemoved = list(set(listOfCandidateCharacters) - set(listOfMatchingChars))
            recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfCandidateCharactersWithCurrentMatchesRemoved)

            for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
                # for each list of matching chars found by recursive call
                # Add lists of matching characters from recursive calls to the main list of lists of matches
                listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

            break

    return listOfListsOfMatchingChars

def getDistanceBetween(charA, charB):
    """
    :param charA:
    :param charB:
    :return: Distance between charA and charB using Pythagorean theorem
    """
    intX = abs(charA.intCenterX - charB.intCenterX)
    intY = abs(charA.intCenterY - charB.intCenterY)
    return math.sqrt((intX ** 2) + (intY ** 2))

def filterCandidateTextRegions(candidateTextRegions):
    """
    Filters out candidate text regions that do not contain characters.
    :param candidateTextRegions:
    :return:
    """
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(candidateTextRegions) == 0:
        return candidateTextRegions

    for candidateRegion in candidateTextRegions:

        candidateRegion.imgGrayscale, candidateRegion.imgThresh = ImageHelper.preprocessNaturalImage(candidateRegion.imgText)
        candidateRegion.imgThresh = cv2.resize(candidateRegion.imgThresh, (0, 0), fx = 1.6, fy = 1.6) # resize

        listPossibleCharsInTextRegion = __findPossibleCharsInTextRegion(candidateRegion.imgGrayscale, candidateRegion.imgThresh)

        # Given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInTextRegion = findListOfListsOfMatchingChars(listPossibleCharsInTextRegion)

        candidateRegion.strChars = ""
        if (len(listOfListsOfMatchingCharsInTextRegion) > 0):
            # Within each list of matching chars, sort chars from L to R
            for i in range(0, len(listOfListsOfMatchingCharsInTextRegion)):
                listOfListsOfMatchingCharsInTextRegion[i].sort(key = lambda matchingChar: matchingChar.intCenterX)

            intLenOfLongestListOfChars = 0
            intIndexOfLongestListOfChars = 0

            # Loop through all the vectors of matching chars, get the index of the one with the most chars
            for i in range(0, len(listOfListsOfMatchingCharsInTextRegion)):
                if len(listOfListsOfMatchingCharsInTextRegion[i]) > intLenOfLongestListOfChars:
                    intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInTextRegion[i])
                    intIndexOfLongestListOfChars = i

            # Suppose that the longest list of matching chars within the plate is the actual list of chars
            longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInTextRegion[intIndexOfLongestListOfChars]

            kNearest = Trainer.getTrainedKNN()
            candidateRegion.strChars = __recognizeCharactersInTextRegion(kNearest, candidateRegion.imgThresh, longestListOfMatchingCharsInPlate)

    return candidateTextRegions

#########################
## PRIVATE METHODS
#########################
def __findPossibleCharsInTextRegion(imgGrayscale, imgThresh):
    """
     Find all contours that could be characters based on area, width, height, and aspect ratio of the contour
     bounding rectangle
    :param imgGrayscale:
    :param imgThresh:
    :return: listOfPossibleChars
    """
    listOfPossibleChars = []

    imgThreshCopy = imgThresh.copy()

    contours = []
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        candidateCharacter = TextRegionFinder.CandidateCharacter(contour)

        if (checkIfCandidateCharacter(candidateCharacter)):
            listOfPossibleChars.append(candidateCharacter)

    return listOfPossibleChars


def __findListOfMatchingChars(candidateChar, listOfChars):
    """
    TODO: Figure out the name of this algorithm or the paper behind it
    Given a possible character and a big list of possible characters, find all the characters in the big list that
    are a possible match for the single possible character, and return matching characters as a list
    :param candidateChar:
    :param listOfChars:
    :return: listOfMatchingChars A list of CandidateCharacter that match
    """
    listOfMatchingChars = []

    for charToMatch in listOfChars:
        if charToMatch != candidateChar:

            # Distance between characters
            fltDistanceBetweenChars = getDistanceBetween(charToMatch, candidateChar)

            # Angle between characters
            fltAdj = float(abs(charToMatch.intCenterX - candidateChar.intCenterX))
            fltOpp = float(abs(charToMatch.intCenterY - candidateChar.intCenterY))

            if fltAdj != 0.0:
                fltAngleInRad = math.atan(fltOpp / fltAdj)
            else:
                # default if adjacent is 0, to avoid divide-by-zero errors
                fltAngleInRad = 1.5708

            fltAngleBetweenChars = fltAngleInRad * (180.0 / math.pi) # degrees

            # Area, Width, Height
            fltChangeInArea = float(abs(charToMatch.intBoundingRectArea - candidateChar.intBoundingRectArea)) / float(candidateChar.intBoundingRectArea)
            fltChangeInWidth = float(abs(charToMatch.intBoundingRectWidth - candidateChar.intBoundingRectWidth)) / float(candidateChar.intBoundingRectWidth)
            fltChangeInHeight = float(abs(charToMatch.intBoundingRectHeight - candidateChar.intBoundingRectHeight)) / float(candidateChar.intBoundingRectHeight)

            if (fltDistanceBetweenChars < (candidateChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
                fltChangeInArea < MAX_CHANGE_IN_AREA and
                fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
                fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
                # Add to list of matching characters if matching criteria met
                listOfMatchingChars.append(charToMatch)

    return listOfMatchingChars


def __recognizeCharactersInTextRegion(kNearest, imgThresh, listOfMatchingChars):
    """
    Recognizes characters in a text region
    :param kNearest:
    :param imgThresh:
    :param listOfMatchingChars:
    :return: strChars the chars in the text region
    """
    strChars = ""
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX) # sort chars from left to right

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor) # color threshold image for contour drawing

    for currentChar in listOfMatchingChars:                                         # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, OCR.SCALAR_GREEN, 2)

        # Crop char out of threshold image and resize
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))

        # Convert image into 1D numpy array of floats
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)

        # Find closest matching character and append to string
        retval, npaResults, neigh_resp, dists = kNearest.find_nearest(npaROIResized, k = 1)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        strChars = strChars + strCurrentChar

    return strChars
