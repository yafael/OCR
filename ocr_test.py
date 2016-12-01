# ocr_test.py
# Runs OCR on test images and compares actual output to expected output.
# Contributors: Edrienne

import os
import numpy as np
import ocr

from difflib import SequenceMatcher

# Files and Directory
TEST_DATA_DIR = "./testdata"
TEST_DATA_EXPECTED = './expectedoutput/testdata_expected_output.txt'

HANDWRITTEN_DATA_DIR = "./testdata/handwritten"
HANDWRITTEN_DATA_EXPECTED = './expectedoutput/handwritten_expected_output.txt'

# Flags and Values
showIndividualResults = False
showImages = True
K = 2

def __getSimilarityScore(expected, actual):
    """ Returns the similarity score for two strings."""
    return SequenceMatcher(None, expected, actual).ratio()

def __getListOfSimilarityScores(expected, actual):
    """Given list of expected string output and actual string output, return list of similarity scores"""
    similarityList = []
    for i in range(len(expected)):
        similarity = __getSimilarityScore(expected[i], actual[i])
        similarityList.append(similarity)
    return similarityList

def __getListOfActualOutput(imgFiles):
    """ Run Optical Character Recognition algorithm on a list of images files and return string output as a list. """
    actualList = []
    for f in imgFiles:
        actual = ocr.recognizeCharacters(f)
        actualList.append(actual)
    return actualList

def __printResults(files, expected, actual, similarity):
    """ Show results for each test image. """
    if (showIndividualResults):
        for i in range(len(files)):
            print "\nExpected = %s\nActual = %s \nSimilarity = %f" % (expected[i], actual[i], similarity[i])
    print "\nMean Similarity = %f" % np.mean(similarity)

def __getFilesAndExpectedValues(fileToExpected, dir):
    """ Load files containing test image names and expected output for each test image and return info as lists. """
    files, expected = [], []
    with open(fileToExpected, "r") as filestream:
        for line in filestream:
            row = line.split(",")
            files.append(os.path.join(dir, row[0]))
            expected.append(row[1].replace('\"', '').rstrip())
    return files, expected

def main():
    """ Runs OCR on test images and compares actual output to expected output. """
    ocr.K = K
    ocr.showImages = showImages

    print "K = %d" % ocr.K
    print "#### TYPEWRITTEN DATA ####"
    testdata_files, testdata_expected = __getFilesAndExpectedValues(TEST_DATA_EXPECTED, TEST_DATA_DIR)
    testdata_actual = __getListOfActualOutput(testdata_files)
    testdata_similarity = __getListOfSimilarityScores(testdata_expected, testdata_actual)
    __printResults(testdata_files, testdata_expected, testdata_actual, testdata_similarity)

    print "\n#### HANDWRITTEN DATA ####"
    handwrittendata_files, handwrittendata_expected = __getFilesAndExpectedValues(HANDWRITTEN_DATA_EXPECTED, HANDWRITTEN_DATA_DIR)
    handwrittendata_actual = __getListOfActualOutput(handwrittendata_files)
    handwrittendata_similarity = __getListOfSimilarityScores(handwrittendata_expected, handwrittendata_actual)
    __printResults(handwrittendata_files, handwrittendata_expected, handwrittendata_actual, handwrittendata_similarity)

if __name__ == "__main__":
	main()