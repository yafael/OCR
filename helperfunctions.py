# ====================
# helper functions for training and testing
# ====================

import cv2
import math
import numpy as np


def sortContours(contour):
    """
	Method used to sort contours from left to right, top to bottom
	:param contour:
	:return: Score used to sort contours
	"""
    tolerance_factor = 50
    mom1 = cv2.moments(contour)

    x, y = 0, 0
    if mom1["m00"] != 0:
        x = int(mom1["m10"] / mom1["m00"])
        y = int(mom1["m01"] / mom1["m00"])

    return ((y // tolerance_factor) * tolerance_factor) * 100 + x


def detectTittles(contour1, contour2):
    """
    :param contour1:
    :param contour2:
    :return: Whether two contours are a character with a tittle
    """
    tolerance_factor = 5
    mom1 = cv2.moments(contour1)
    mom2 = cv2.moments(contour2)

    x1, x2 = 0, 0
    if mom1["m00"] != 0:
        x1 = int(mom1["m10"] / mom1["m00"])
        x2 = int(mom1["m01"] / mom1["m00"])

    # y1 = int(mom1['m01'] / mom1['m00'])
    # y2 = int(mom2['m01'] / mom2['m00'])

    if (abs(x1 - x2) < tolerance_factor):
        return True
    else:
        return False

def getDistanceBetween(charA, charB):
    """
    :param charA: Contour for character A
    :param charB: Contour for character B
    :return: Distance between the center of charA and charB using the Pythagorean theorem
    """
    MomCharA = cv2.moments(charA)
    MomCharB = cv2.moments(charB)

    cxA, cYA, cXB, cYB = 0, 0, 0, 0

    if (MomCharA["m00"] != 0):
        cXA = int(MomCharA["m10"] / MomCharA["m00"])
        cYA = int(MomCharA["m01"] / MomCharA["m00"])

    if (MomCharB["m00"] != 0):
        cXB = int(MomCharB["m10"] / MomCharB["m00"])
        cYB = int(MomCharB["m01"] / MomCharB["m00"])

    intX = abs(cXA - cXB)
    intY = abs(cYA - cYB)

    return math.sqrt((intX ** 2) + (intY ** 2))

def getIndexOfTittle(contour, lettersWithTittles):
    """
    :param contour:
    :param lettersWithTittles: 
    :return: i The index of the first contour that has a letter with a tittle
    """
    for i in range(len(lettersWithTittles)):
        if (np.any(contour == lettersWithTittles[i])):
            return i
    return -1

def findValidContours(contours):
    """
    :param contours: list of contours
    :return: list of contours that meet our criteria
    """
    areaList = []
    validList = []

    for i in contours:
        areaList.append(cv2.contourArea(i))

    meanArea = np.mean(areaList)
    stdDev = np.std(areaList)

    print "Mean area = %f" % meanArea
    print "Standard Deviation = %f" % stdDev

    for i in contours:
        if(abs(cv2.contourArea(i) - meanArea) <= 2*stdDev):
            validList.append(i)

    return validList

def findValidRectangles(rects):
    """
    TODO: Delete. This isn't used anywhere.
    :param rects:
    :return: validList of rectangles
    """
    validList = []

    meanArea = np.mean(rects)
    stdDev = np.std(rects)

    print "Mean area = %f" % meanArea
    print "Standard Deviation = %f" % stdDev

    for i in rects:
        if(abs(rects[i] - meanArea) <= stdDev):
            validList.append(i)

    return validList
