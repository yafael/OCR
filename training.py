# Author: Kyla Bouldin
# Description: creates training data file based on arial letter characters
# (10x15 image)


import numpy as np
import cv2

def getTrainingData():
	trainingdata = {}
	alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


	for i in range(len(alphabet)):
		filename = 'trainingdata/%s.png' % alphabet[i]
		letter = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		trainingdata[alphabet[i]] = letter
		#cv2.imshow('image',letter)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
	
	return trainingdata
	
print(getTrainingData())