import os
import numpy as np

from difflib import SequenceMatcher

import test

# Files and Directory
TEST_DATA_DIR = "./testdata"
HANDWRITTEN_DATA_DIR = "./handwrittendata"
TEST_DATA_EXPECTED = './accuracydata/testdata_expected_output.txt'

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    testdata_files, testdata_expected = [], []

    with open(TEST_DATA_EXPECTED, "r") as filestream:
        for line in filestream:
            row = line.split(",")
            testdata_files.append(os.path.join(TEST_DATA_DIR, row[0]))
            testdata_expected.append(row[1].replace('\"', '').rstrip())

    similarityList = []
    for i in range(len(testdata_files)):
        f = testdata_files[i]
        expected = testdata_expected[i]
        actual = test.recognizeCharacters(f)
        similarity = similar(expected, actual)
        similarityList.append(similarity)
        print "[%s] \nExpected = %s\nActual = %s \nSimilarity = %f\n" % (f, expected, actual, similarity)

    meanSimilarity = np.mean(similarityList)
    stdDevSimilarity = np.std(similarityList)
    print "Average Similarity = %f" % meanSimilarity
    print "Standard Deviation Similarity = %f" % stdDevSimilarity

if __name__ == "__main__":
	main()