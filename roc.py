__author__ = 'annapurnaannadatha'

#############################################################################
##                               README                                    ##
##                                                                         ##
##     This program plots ROC curve and compute AUC values. Also prints    ##
##     the ROC points. It takes scores for two different classes as two    ##
##     seperate CSV files.                                                 ##
##                                                                         ##
#############################################################################



from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from matplotlib import style
style.use("ggplot")


# header
data_headers = ['score']

# Reading spam data from csv file into an array
data_df = pd.DataFrame.from_csv("spam_score_n3.csv")
malware_X = np.array(data_df[data_headers].values)
print("No of malware files with no of features:",malware_X.shape)
# classifying ham data as 1
test_mal_category = np.empty([len(malware_X),1])
for i in range(0,len(malware_X)):
    test_mal_category[i]= -1

#Reading benign data from csv file into an array
data1_df = pd.DataFrame.from_csv("ham_score_n3.csv")
benign_X = np.array(data1_df[data_headers].values)
print("No of benign files with no of features:",benign_X.shape)
# classifying benign data as -1
test_ben_category = np.empty([len(benign_X),1])
for i in range(0,len(benign_X)):
    test_ben_category[i]= 1


# merging benign and malware data into test_set and train_set
test_set = np.concatenate((malware_X,benign_X[:len(malware_X)]))
test_out = np.concatenate((test_mal_category,test_ben_category[:len(malware_X)]))


print("test out size with number of features:",test_out.shape)
print("test set size with number of features:",test_set.shape)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(test_out, test_set)
roc_auc = auc(fpr, tpr)

for i in range(len(fpr)):
    print(fpr[i],"\t",tpr[i])

##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr,color ='b', label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic ')
plt.legend(loc="lower right")
plt.show()


