import cv2

def sortContours(point1):
    tolerance_factor = 60
    mom1 = cv2.moments(point1)

    x = int(mom1['m10'] / mom1['m00'])
    y = int(mom1['m01'] / mom1['m00'])

    return ((y // tolerance_factor) * tolerance_factor) * 100 + x