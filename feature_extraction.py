__author__ = 'annapurnaannadatha'

#############################################################################
##                               README                                    ##
##                                                                         ##
##     This program extracts certain features of an image file. Numpy      ##
##     array is used to store the image pixel values. Each feature         ##
##     extracted has a description below answering 'Which' feature is      ##
##     extracted, 'What'does it denote, 'Why' is it extracted and 'How'    ##
##     the feature is extracted. The extracted features are written        ##
##     to a csv file for further classification of images as SPAM or       ##
##     HAM(Genuine image, just to rhyme with spam) images. Also prints     ##
##     the time taken for feature extraction.                              ##
##     The path to the folder of images has to be changed accordingly...   ##
##                                                                         ##
##                                                                         ##
#############################################################################

import numpy as np
import cv2
import math
import os
import csv
from PIL import Image
from skimage.color import rgb2gray
from multiprocessing import Pool
from skimage.feature import local_binary_pattern # Local Binary Pattern function
from scipy.stats import itemfreq # To calculate a normalized histogram
import scipy.stats as sp
from skimage.feature import hog
from scipy.ndimage.measurements import label
from scipy import signal as sg
import time # tocheck time for feature extraction

#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Compression Ratio (Metadata domain)
## What  : Ratio of Size of the original image to the changed image
## Why   : Spammers might try to compress image as far as possible.
##         Thus could show difference between spam and ham images
##         Spam image generally has high compression rate
## How   : (height * width * bitdepth) / Size of the image
##
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_compression_ratio(img,image,file):
    height, width ,channels= img.shape
    file_size = os.stat(path1 + file).st_size
    depth = image.bits * channels
    compression_ratio = (height * width * depth )/file_size
    return compression_ratio


#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Aspect Ratio (Metadata domain)
## What  : Ratio of Size of the original image to the changed image
## Why   : Spammers might try to compress image as far as possible.
##         Thus could show difference between spam and ham images
##         Spam image generally has high compression rate
## How   : (height * width * bitdepth) / Size of the image
##
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_aspect_ratio(img):
    height, width ,channels= img.shape
    aspect_ratio = width/height
    return aspect_ratio




#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Number of Edge pixels(Shape domain)
## What  : Edges is the points in a image at which image brightness
##         changes sharpely or has discontinuities and are organized
##         into a set of curved line segments
## Why   : Spam images are combinations of text and images, they are
##         likely to have more number of edges compared to genuine images
## How   : Identifying the edges using Canny edge detection
##         method and counting the number of edges
##
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_edge_count(img):
    edges = cv2.Canny(img,100,200)
    edge_count = np.count_nonzero(edges)
    return edge_count


#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Average length of edges(Shape domain)
## What  : Edges is the points in a image at which image brightness
##         changes sharpely or has discontinuities and are organized
##         into a set of curved line segments
## Why   : Spam images are combinations of text and images, they are
##         likely to have more number of edges compared to genuine images
## How   : Identifying the edges using Canny edge detection
##         method and counting the number of edges
##
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_avg_edge_length(img):
    edges = cv2.Canny(img,100,200)
    length = sum(map(sum, edges))
    labeled_array, num_features = label(edges)
    avg_edge_len= length/num_features
    return avg_edge_len






#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Signal to Noise Ratio (SNR)(Noise domain)
## What  : SNR is used as a measure of the sensitivity of a image.
## Why   : Genuine images have significant noise components
## How   : Ratio of Mean to the standard deviation
##
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_snr(img):
    snr = sp.signaltonoise(img, axis=None)
    return snr

#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Entropy of Noise(Noise domain)
## What  : The LBP contained in the image reveals the similarity of
##         differences between the neighboring pixels and can be used
##         to analyse texture information
## Why   : Genuine images are likely to have more information contained
##         in the LBP histogram.Spam images are likely to have more
##         smoother background and foreground features
## How   :
##
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_entropy_noise(img):
    img_gray = rgb2gray(img)
    H, W = img_gray.shape

    M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(sg.convolve2d(img_gray, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma



#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Entropy of Local Binary Pattern Histogram(Texture domain)
## What  : The LBP contained in the image reveals the similarity of
##         differences between the neighboring pixels and can be used
##         to analyse texture information
## Why   : Genuine images are likely to have more information contained
##         in the LBP histogram.Spam images are likely to have more
##         smoother background and foreground features
## How   :
##
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_entropy_lbp(img):
    img_gray = rgb2gray(img)
    radius = 3
    # Number of points to be considered as neighbours
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(img_gray, no_points, radius, method='uniform')
    # Calculate the histogram
    x = itemfreq(lbp.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    lbp_entropy = -sum([p * math.log(p, 2) for p in hist if p != 0])
    return lbp_entropy


#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Entropy of Color histogram(Color domain)
## What  : Representation of distribution of colors in an image(RGB)
## Why   : Genuine images are supposed to have more information contained in
##         colors as  they would have been captured by camera
## How   : Computing the number of pixels that have colors in each of a
##         fixed list of color ranges
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_entropy_color_hist(img):
    cal_hist = cv2.calcHist([img], [0, 1, 2],None, [10, 10, 10], [0, 256, 0, 256, 0, 256])
    # Calculate the histogram
    y = itemfreq(cal_hist.ravel())
    # Normalize the histogram
    hist_color = y[:, 1]/sum(y[:, 1])
    # entropy
    entropy_color = -sum([p * math.log(p, 2) for p in hist_color if p != 0])
    return entropy_color


############################################################################
##                                                                         ##
##                                                                         ##
## Which : Entropy of Oriented Gradient Histogram(Color domain)
## What  : Representation of distribution of colors in an image(RGB)
## Why   : Genuine images are supposed to have more information contained in
##         orientations
## How   : Compute the edge orientations
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_entropy_hog(img):
    img_gray = rgb2gray(img)
    fd, hog_image = hog(img_gray, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
    # Calculate the histogram
    x = itemfreq(hog_image.ravel())
    # Normalize the histogram
    hist = x[:, 1]/sum(x[:, 1])
    entropy_hog = -sum([p * math.log(p, 2) for p in hist if p != 0])
    return entropy_hog

#############################################################################
##                                                                         ##
##                                                                         ##
## Which : Mean,Variance,Kurtosis,Skewness of RGB channels
## What  : Representation of distribution of colors in an image(RGB)
## Why   : Genuine images are supposed to have more information contained in
##         colors as  they would have been captured by camera
## How   : Computing the number of pixels that have colors in each of a
##         fixed list of color ranges
##                                                                         ##
##                                                                         ##
#############################################################################
def calc_channel_hist(img):
    chans = cv2.split(img)
    #print(chans)
    mean=[]
    kurtosis=[]
    variance=[]
    skew=[]
    for chan in chans:
        # Calculate the histogram
        histo = cv2.calcHist([chan],[0],None,[100],[0,256])
        # Normalize the histogram
        hist_length = sum(histo)
        hist = [float(h) / hist_length for h in histo]
        #print(ss.describe(hist))
        skew.append(sp.skew(hist)[0])
        kurtosis.append(sp.kurtosis(hist)[0])
        mean.append(sp.tmean(hist))
        variance.append(sp.tvar(hist))
    return mean,variance,kurtosis,skew






# opening csv file to write the output
c = csv.writer(open("spam_ds3.csv",'w',encoding='utf8',newline=''))

# title line for the csv file
c.writerow(('Image_name','compression_ratio','aspect ratio','edge count','Avg edge length','SNR','Entropy of noise','Entropy of LBP','Entropy of color histogram','Entropy of HOG','Mean1','Mean2','Mean3','Variance1','Variance2','Variance3','Skew1','Skew2','Skew3','kurtosis1','kurtosis2','kurtosis3'))
#c.writerow(('Image_name','SNR','Entropy of noise','Entropy of LBP','Entropy of HOG','Variance1','Variance2','Variance3'))


# image folder path
path1 = "dredze_spam/"

# list to store file names of the folder
listing = os.listdir(path1)

#process n images simulateously to decrease the processing time
#p = Pool(10)
# Mapping to the extract_feature function with listing
#p.map(extract_feature, listing)

print()
print( "Extracting features for following images.... ")

# check the start time
start=time.time()

for file in listing:
    # to avoid .ds_store file in mac
    if not (file.startswith('.')):
        # reading each image file
        img = cv2.imread(path1 + file)
        image = Image.open(path1 + file)

        print(file)

        # compression ratio
        compression_ratio = calc_compression_ratio(img,image,file)
        # aspect ratio
        aspect_ratio = calc_aspect_ratio(img)
        # number of edges
        edge_count = calc_edge_count(img)
        # Average length of edges
        avg_edge_len = calc_avg_edge_length(img)
        # signal to noise ratio
        snr = calc_snr(img)
        # Entropy of noise
        noise_entropy = calc_entropy_noise(img)
        # Entropy of LBP histogram
        lbp_entropy = calc_entropy_lbp(img)
        # Entropy of color histogram
        entropy_color = calc_entropy_color_hist(img)
        # Entropy of Oriented Gradient Histogram (HOG)
        entropy_hog = calc_entropy_hog(img)
        # Mean,variance,kurtosis,skewness of RGB channels
        mean,var,kurt,skew = calc_channel_hist(img)
        # writing data to csv file
        c.writerow((file,compression_ratio,aspect_ratio,edge_count,avg_edge_len,snr,noise_entropy,lbp_entropy,entropy_color,entropy_hog,mean[0],mean[1],mean[2],var[0],var[1],var[2],skew[0],skew[1],skew[2],kurt[0],kurt[1],kurt[2]))


#check the end time
end=time.time()

print()
print("Done ... writing file ")
print()
print('Time taken in seconds:',end-start)

exit()