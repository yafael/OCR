# ====================
# helper functions for training and testing
# ====================

import cv2
import math
import numpy as np

def sortContours(point1):
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
	
def detectTittles(point1, point2):
    tolerance_factor = 5
    mom1 = cv2.moments(point1)
    mom2 = cv2.moments(point2)

    x1 = int(mom1['m10'] / mom1['m00'])
    x2 = int(mom2['m10'] / mom2['m00'])

    #y1 = int(mom1['m01'] / mom1['m00'])
    #y2 = int(mom2['m01'] / mom2['m00'])

    if (abs(x1 - x2) < tolerance_factor):
        return True
    else:
        return False

def getDistanceBetween(charA, charB):
	MomCharA = cv2.moments(charA)
	MomCharB = cv2.moments(charB)
	cXA = int(MomCharA["m10"] / MomCharA["m00"])
	cYA = int(MomCharA["m01"] / MomCharA["m00"])
	cXB = int(MomCharB["m10"] / MomCharB["m00"])
	cYB = int(MomCharB["m01"] / MomCharB["m00"])
	intX = abs(cXA - cXB)
	intY = abs(cYA - cYB)
	return math.sqrt((intX ** 2) + (intY ** 2))
	

def getIndexOfTittle(contour, letterWithTittle):
	for i in range(len(letterWithTittle)):
		if (np.any(contour == letterWithTittle[i])):
			return i
	return -1