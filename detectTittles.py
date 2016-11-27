import cv2


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
