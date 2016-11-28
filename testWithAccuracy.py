import os
import numpy as np

from difflib import SequenceMatcher

import test

# Files and Directory
TEST_DATA_DIR = "./testdata"
TEST_DATA_EXPECTED = './accuracydata/testdata_expected_output.txt'

HANDWRITTEN_DATA_DIR = "./handwrittendata"
HANDWRITTEN_DATA_EXPECTED = './accuracydata/handwritten_expected_output.txt'

# Flags
showIndividualResults = False

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def getListOfSimilarityScores(files, expected, actual):
    similarityList = []
    for i in range(len(files)):
        similarity = similar(expected[i], actual[i])
        similarityList.append(similarity)
    return similarityList


def getListOfActualOutput(imgFiles):
    actualList = []
    for f in imgFiles:
        actual = test.recognizeCharacters(f)
        actualList.append(actual)
    return actualList

def printResults(files, expected, actual, similarity):
    if (showIndividualResults):
        for i in range(len(files)):
            print "[%s] \nExpected = %s\nActual = %s \nSimilarity = %f\n" % (files[i], expected[i], actual[i], similarity[i])
    print "Average Similarity = %f" % np.mean(similarity)
    print "Standard Deviation Similarity = %f" % np.std(similarity)

def getFilesAndExpectedValues(fileToExpected, dir):
    files, expected = [], []
    with open(fileToExpected, "r") as filestream:
        for line in filestream:
            row = line.split(",")
            files.append(os.path.join(dir, row[0]))
            expected.append(row[1].replace('\"', '').rstrip())
    return files, expected

def main():
    # TEST DATA

    testdata_files, testdata_expected = getFilesAndExpectedValues(TEST_DATA_EXPECTED, TEST_DATA_DIR)
    testdata_actual = getListOfActualOutput(testdata_files)
    testdata_similarity = getListOfSimilarityScores(testdata_files, testdata_expected, testdata_actual)
    print "#### TYPEWRITTEN DATA ####"
    printResults(testdata_files, testdata_expected, testdata_actual, testdata_similarity)

    # HANDWRITTEN DATA
    handwrittendata_files, handwrittendata_expected = getFilesAndExpectedValues(HANDWRITTEN_DATA_EXPECTED, HANDWRITTEN_DATA_DIR)
    handwrittendata_actual = getListOfActualOutput(handwrittendata_files)
    handwrittendata_similarity = getListOfSimilarityScores(handwrittendata_files, handwrittendata_expected, handwrittendata_actual)
    print "\n#### HANDWRITTEN DATA ####"
    printResults(handwrittendata_files, handwrittendata_expected, handwrittendata_actual, handwrittendata_similarity)


if __name__ == "__main__":
	main()