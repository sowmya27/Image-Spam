__author__ = 'annapurnaannadatha'

#############################################################################
##                               README                                    ##
##                                                                         ##
##     This program plots the probability distributions for each feature.  ##
##     It takes feature values for two different classes as seperate CSV   ##
##     files.                                                              ##
##                                                                         ##
#############################################################################

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import style
style.use("ggplot")



data_headers = ['compression_ratio','aspect ratio','edge count','Avg edge length','SNR','Entropy of noise','Entropy of LBP','Entropy of color histogram','Entropy of HOG','Mean1','Mean2','Mean3','Variance1','Variance2','Variance3','Skew1','Skew2','Skew3','kurtosis1','kurtosis2','kurtosis3']

# Reading malware data from csv file into an array
data_df = pd.DataFrame.from_csv("spam_my.csv")
malware_X = np.array(data_df[data_headers].values)
print("No of malware files with no of features:",malware_X.shape)

#Reading benign data from csv file into an array
data1_df = pd.DataFrame.from_csv("ham_my.csv")
benign_X = np.array(data1_df[data_headers].values)
print("No of benign files with no of features:",benign_X.shape)


# Feature number ( could be done as for loop too but inorder to change the bins plotted seperately)
Fno = 0

x_1 = sorted(malware_X[:,Fno])
x_2 = sorted(benign_X[:,Fno])

#normal distribution

fit_1 = stats.norm.pdf(x_1, np.mean(x_1), np.std(x_1))
fit_2 = stats.norm.pdf(x_2, np.mean(x_2), np.std(x_2))

#bins=(0,1,2,3,4,5,6,7,8,9,10) [:,4]
#bins = range(0,10) [:4]
#bins = range(0,200) [:0]
#bins=[0,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000] [:2]
# no bins [:6] [:8] [:5] [:3]

plt.hist(x_1,label='Spam Image',color='g')
plt.hist(x_2,color='b',label='Ham Image')

plt.title(data_headers[Fno])
plt.xlabel(data_headers[Fno])
plt.ylabel("Frequency")
plt.legend(loc='upper right')
plt.gca().grid(False)
plt.show()


plt.plot(x_1,fit_1,'r',label='Malware')
plt.plot(x_2,fit_2,'b',label='benign')
plt.title(data_headers[Fno])
plt.xlabel(data_headers[Fno])
plt.ylabel("Probability")
plt.legend(loc='upper right')
plt.gca().grid(False)
plt.show()



