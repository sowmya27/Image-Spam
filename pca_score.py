__author__ = 'annapurnaannadatha'


#############################################################################
##                               README                                    ##
##                                                                         ##
##     This program performs Principal Component Analysis and generates    ##
##     a model by training Spam images. The images are first preprocessed  ##
##     so that all images are of same size and convereted to grayscale.    ##  
##     PCA scores are computed for the test images (Ham and Spam). CSV     ##
##     files are generated with the scores seperately for spam and ham     ##
##     images.                                                             ##
##                                                                         ##
##                                                                         ##
##                                                                         ##
#############################################################################

from sklearn.decomposition import PCA
import numpy
import glob
import cv2
import csv
import math
import os
import string
from scipy.misc import *
from scipy import linalg
from skimage.color import rgb2gray
from PIL import Image


#function to convert image to right form: resize , convert to grayscale and flatten
def prepare_image(filename):
    #print(filename)
    img_color = cv2.imread(filename)
    image = cv2.resize(img_color,(100,100),interpolation = cv2.INTER_AREA)
    img_gray = rgb2gray(image)
    return img_gray.flatten()

#Score
def score(path,csvfile):
    # title line for the csv file
    csvfile.writerow(('value','score'))
    test_listing = os.listdir(path)
    #test files
    for file in test_listing:
        Y = numpy.array([prepare_image(path + file) ])
        # run through test images
        print(file)
        for j, ref_pca in enumerate(pca.transform(Y)):
            i=0
            distances = []
            # Calculate euclidian distance from test image to each of the known images and save distances
            for i, test_pca in enumerate(X_pca):
                dist = math.sqrt(sum([diff**2 for diff in (ref_pca - test_pca)]))
                distances.append((dist,y[i]))
            found_ID = min(distances)[1]
            print("Identified (result: "+ str(found_ID) +" - dist - " + str(min(distances)[0])  + ")")
            csvfile.writerow(("no", round(min(distances)[0],4)))

#image folder path
path_in = "spam_train/"
spam_test_path = "spam_test/"
ham_test_path = "ham/"

# list to store file names of the folder
listing = os.listdir(path_in)

#populate into each image into array
X = numpy.array([prepare_image(path_in + file) for file in listing])
y = []
for file in listing:
    y.append(file)

# perform principal component analysis on the images
pca = PCA(whiten=True).fit(X)
X_pca = pca.transform(X)

# opening csv file to write the output
c = csv.writer(open("spam_score_n300.csv",'w',encoding='utf8',newline=''))
c1 = csv.writer(open("ham_score_n300.csv",'w',encoding='utf8',newline=''))

# scoring phase
score(ham_test_path,c1)
score(spam_test_path,c)