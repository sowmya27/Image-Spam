__author__ = 'annapurnaannadatha'
#############################################################################
##                               README                                    ##
##                                                                         ##
##     This program generates a scatterplot. It takes scores for two       ##
##     different classes as two seperate CSV files.                        ##
##                                                                         ##
#############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# listing the features included
data_headers = ['score']

# Reading malware data from csv file into an array
data_df = pd.DataFrame.from_csv("spam_score_n3.csv")
malware_X = np.array(data_df[data_headers].values)
print("No of malware files with no of features:",malware_X.shape)
# classifying malware data as 1
indices_malware = np.empty([len(malware_X),1])
for i in range(0,len(malware_X)):
    indices_malware[i]= i

#Reading benign data from csv file into an array
data1_df = pd.DataFrame.from_csv("ham_score_n3.csv")
benign_X = np.array(data1_df[data_headers].values)
print("No of benign files with no of features:",benign_X.shape)
# classifying benign data as -1
indices_benign = np.empty([len(benign_X),1])
for i in range(0,len(benign_X)):
    indices_benign[i]= i
benign_X = benign_X
indices_benign = indices_benign

for i in range(len(malware_X)):
    print(str(i),"\t",benign_X[i][0],"\t",malware_X[i][0])
#print("No of benign files with no of features:",benign_X.shape)

plt.figure()
spam = plt.scatter(indices_malware, malware_X, color='r',label='Spam Image')
ham = plt.scatter(indices_benign[:len(malware_X)], benign_X[:len(malware_X)], color='b',label='Ham Image')
plt.legend((spam,ham),
           ('Spam Image','Ham Image'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
plt.xlabel('No. of Images')
plt.ylabel('Scores')
plt.show()


'''
ax.scatter(x1, y1, color='r', s=2*s, marker='^', alpha=.4)
ax.scatter(x2, y2, color='b', s=s/2, alpha=.4)
ax.scatter(x3, y3, color='g', s=s/3, marker='s', alpha=.4)
'''