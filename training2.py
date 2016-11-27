# Author: Kyla Bouldin, Yashvardhan Gusani
# Description: creates training data and classification files

import sys
import numpy as np
import cv2

# constants
MIN_CONTOUR_AREA = 10
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


# ====================
# Custom function designed to sort the contours
# ====================
def sortContours(point1):
    tolerance_factor = 60
    mom1 = cv2.moments(point1)

    x = int(mom1['m10'] / mom1['m00'])
    y = int(mom1['m01'] / mom1['m00'])

    return ((y // tolerance_factor) * tolerance_factor) * 100 + x


# ====================
# main function; starts training process
# ====================
def main():
    # read in training numbers image, grayscale it, filter to black and white, display image, make a copy
    imgTrainingNumbers = cv2.imread("training_chars.png")
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imshow("imgThresh", imgThresh)
    imgThreshCopy = imgThresh.copy()

    # find and sort contours
    contours, hierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=lambda x: sortContours(x))

    # declare empty numpy array for each training data sample
    trainingdata = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    # declare empty classifications list
    intClassifications = []

    # show found contours using bounding rect
    for contour in contours:
        if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(contour)
            cv2.rectangle(imgTrainingNumbers, (intX, intY), (intX + intW, intY + intH), (255, 0, 255), 1)

            # USE THIS WHEN FINDING CONTOURS ON HANDWRITTEN IMAGE TOO
            contourImg = imgThresh[intY:intY + intH, intX:intX + intW]
            contourImgResized = cv2.resize(contourImg, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("training_numbers.png", imgTrainingNumbers)
            sampleLetter = contourImgResized.reshape((1,
                                                      RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
            trainingdata = np.append(trainingdata, sampleLetter,
                                     0)  # add current flattened impage numpy array to list of flattened image numpy arrays

    cv2.waitKey(0)

    # create classifications - has to be a number
    intClassifications = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'),
                          ord('9'),
                          ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'),
                          ord('J'),
                          ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'),
                          ord('T'),
                          ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    # convert classifications list of ints to numpy array of floats
    floatClassifications = np.array(intClassifications, np.float32)

    # flatten to 1d
    flatClassifications = floatClassifications.reshape((floatClassifications.size, 1))

    print "\n\ntraining complete !!\n"

    np.savetxt("classification_labels.txt", flatClassifications)  # write flattened images to file
    np.savetxt("training_data.txt", trainingdata)  #

    cv2.destroyAllWindows()  # remove windows from memory

    return


# ====================
# Run main methods
# ====================
main()

