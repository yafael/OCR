# Main Contributor: Edrienne
# Description: Main class for Optical Character Recognition.
# TODO:

import cv2
import numpy as np
import os

import CharacterHelper
import Trainer
import TextRegionFinder

# Constants
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

def __readNaturalImage(imgFilename):
    """
    Reads image from filename, prints error if doesn't exist
    :param imgFilename:
    :return:
    """
    img = cv2.imread(imgFilename)
    if img is None:
        print "\nerror: image not read from file \n\n"
        os.system("pause")
        return
    return img

def __boxTextRegion(img, textRegion):
    """
    Adds red box around a text region
    :param img:
    :param textRegion:
    :return: void
    """
    boxPoints = cv2.cv.BoxPoints(textRegion.rrLocationOfTextInScene)
    box = np.int0(boxPoints)
    cv2.drawContours(img, [box], 0, SCALAR_RED, 2)

def main():
    """
    Trains character data and then runs Optical Character Recognition on natural images.
    :return:
    """
    # Trainer.trainData("training_letters.png")
    # ISSUE: For all programs, not just mine. Training data produces blank
    # classifications.txt and flattened_images.txt files
    
    img = __readNaturalImage("1.png")
    candidateTextRegions = TextRegionFinder.getCandidateTextRegions(img)
    candidateTextRegions = CharacterHelper.filterCandidateTextRegions(candidateTextRegions)

    if len(candidateTextRegions) == 0:
        print "\nNo text detected.\n"
    else:
        # Process all plates that have at least one character
        i = 1;
        for textRegion in candidateTextRegions:
            str = "No Characters Found"
            if len(textRegion.strChars) > 0:
                __boxTextRegion(img, textRegion)  # draw red rectangle around plate
                str = textRegion.strChars

            print "Text Region %d: %s" % (i, str)
            i = i + 1

        cv2.imshow("Image with Text Regions", img)
        cv2.imwrite("img.png", img)

    cv2.waitKey(0)					# hold windows open until user presses a key

    return

if __name__ == "__main__":
    main()














