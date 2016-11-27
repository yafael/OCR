import cv2

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