__author__ = 'annapurnaannadatha'

#############################################################################
##                               README                                    ##
##                                                                         ##
##     This program classifies SPAM and HAM images using SVM (Support      ##
##     Vector machine) technique. The image features values for the        ##
##     malware and benign are used to train and test the data. It takes    ##
##     image feature values for two classes( ham.csv, spam.csv). These     ##
##     files are output of the feature extraction module.                  ##
##                                                                         ##
##                                                                         ##
#############################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import svm, preprocessing
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import style
style.use("ggplot")
import warnings  # to ignore all the deprecated warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
   # listing the features included
   data_headers = ['compression_ratio','aspect ratio','edge count','Avg edge length','SNR','Entropy of noise','Entropy of LBP','Entropy of color histogram','Entropy of HOG','Mean1','Mean2','Mean3','Variance1','Variance2','Variance3','Skew1','Skew2','Skew3','kurtosis1','kurtosis2','kurtosis3']

   # Reading malware data from csv file into an array
   data_df = pd.DataFrame.from_csv("spam_ds3.csv")
   malware_X = np.array(data_df[data_headers].values)
   print("No of malware files with no of features:",malware_X.shape)
   # classifying malware data as 1
   test_mal_category = np.empty([len(malware_X),1])
   for i in range(0,len(malware_X)):
       test_mal_category[i]= 1

   #Reading benign data from csv file into an array
   data1_df = pd.DataFrame.from_csv("ham_ds3.csv")
   benign_X = np.array(data1_df[data_headers].values)
   print("No of benign files with no of features:",benign_X.shape)
   # classifying benign data as -1
   test_ben_category = np.empty([len(benign_X),1])
   for i in range(0,len(benign_X)):
       test_ben_category[i]= -1

   # test size is set as 100 for each of benign and malware, so total test set include 200 files
   test_size = 400

   # merging benign and malware data into test_set and train_set
   test_set = np.concatenate((malware_X[:test_size],benign_X[:test_size]))
   train_set = np.concatenate((malware_X[test_size:],benign_X[test_size:]))

   print("train set size with number of features:",train_set.shape)
   print("test set size with number of features:",test_set.shape)

   # merging the out(classification) data for test set and train set
   test_out = np.concatenate((test_mal_category[:test_size],test_ben_category[:test_size]))
   train_out = np.concatenate((test_mal_category[test_size:],test_ben_category[test_size:]))

   # preprocessing and scaling of the data
   train_set = preprocessing.scale(train_set)
   test_set = preprocessing.scale(test_set)

   print("######################################################################")

   ## classification
   classification("rbf",test_set,train_set,test_out,train_out)
   classification("linear",test_set,train_set,test_out,train_out)
   classification("poly",test_set,train_set,test_out,train_out)

   ## compute svm weights
   svm_weights = compute_svm_weights(train_set,train_out)
   plot_hist_weights(train_set,svm_weights)

   ## RFE feature selection with number of features to be selected
   i = 12
   ## RFE classification
   rfe_classification("rbf",i,test_set,train_set,test_out,train_out)
   rfe_classification("linear",i,test_set,train_set,test_out,train_out)
   rfe_classification("poly",i,test_set,train_set,test_out,train_out)


## SVM classification
def classification(kernel,test_set,train_set,test_out,train_out):
    #kernel
    clf = svm.SVC(kernel= kernel, C= 1.00)
    print("---------------------"  + kernel +" Kernel----------------------")
    clf_model= clf.fit(train_set,train_out.ravel())
    predicted = clf.predict(test_set)
    print_evaluation(predicted,test_out)
    ## ROC curveeee
    compute_roc_curve(kernel,21,clf_model,test_set,train_set,test_out,train_out)
    print("######################################################################")

## To print roc curve  points
def roc_points_print(fpr,tpr):
    for i in range(len(fpr)):
        print(fpr[i],"\t",tpr[i])

## Plots roc curve and prints AUC value
def compute_roc_curve(kernel,i,model,test_set,train_set,test_out,train_out):
    # To compute and plot ROC curve
    y_score = model.decision_function(test_set)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(test_out, y_score)
    roc_auc = auc(fpr, tpr)
    print("AUC = %0.2f"%roc_auc)
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr,color ='b', label='AUC with %d features = %0.2f'%( i,roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(kernel)
    plt.legend(loc="lower right")
    plt.show()
    #print
    #roc_points_print(fpr,tpr)

## Computes Linear SVM weights 
def compute_svm_weights(train_set,train_out):
    # svm weights for linear kernel only
    clf = svm.SVC(kernel="linear", C= 1.00)
    clf.fit(train_set,train_out.ravel())
    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= svm_weights.max()
    #print(svm_weights)
    return svm_weights

## PLots histogram of the SVM wieghts
def plot_hist_weights(train_set,svm_weights):
    X_indices = np.arange(train_set.shape[-1])
    #  plotting histogram for the svm weights of each feature
    plt.bar(X_indices - .45, svm_weights, width=0.5, label='SVM weight', color='r')
    plt.title("SVM weights")
    plt.xlabel('Feature')
    plt.yticks(())
    plt.xticks(X_indices,('comp','aspect','edges','edgelen','SNR','noise','LBP','color_hist','HOG','Mean1','Mean2','Mean3','Var1','Var2','Vari3','Skew1','Skew2','Skew3','kurt1','kurt2','kurt3'))
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()
    '''
    # plotting rfe algorithm ranks on the same histogram of svm weights for comparision
    plt.bar(X_indices - .05, rfe.support_, width=.5, label='RFE Selection', color='b')
    plt.legend(loc='upper right')
    plt.axis('tight')
    plt.show()
    '''

## RFE selection and classification based no features selected
def rfe_classification(kernel,i,test_set,train_set,test_out,train_out):
    # Recursive feature elimation
    model = svm.SVC(kernel="linear", C=1.00) #LogisticRegression()
    rfe = RFE(model,i) # no of features to be selected
    rfe = rfe.fit(train_set,train_out.ravel())
    rfe_weights = (rfe.support_ ** 2).sum(axis=0)
    rfe_weights /= rfe_weights.max()
    # Transform the featureset from 21 to subset (i)  as selected in the RFE computation above
    rfe_train = rfe.transform(train_set)
    rfe_test = rfe.transform(test_set)
    clf = svm.SVC(kernel=kernel, C= 1.00)
    print("---------------------RFE : "+kernel+" Kernel----------------------")
    rfe_model = clf.fit(rfe_train,train_out.ravel())
    rbf_predict = clf.predict(rfe_test)
    print_evaluation(rbf_predict,test_out)
    #roc
    compute_roc_curve(kernel,i,rfe_model,rfe_test,rfe_train,test_out,train_out)
    print("######################################################################")


## Prints the final evaluation
def print_evaluation(predict,test_out):
    conf_mat = confusion_matrix(test_out,predict)
    print(conf_mat)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    print("Accuracy= %0.2f"%Accuracy)
    TPR = TP/(TP + FN)
    print("Recall TPR=%0.2f"%TPR)
    FPR = FP/(FP + TN)
    print("FPR=%0.2f"%FPR)
    PPR = TP/(TP +FP)
    print("Precision PPR=%0.2f"%PPR)



## Main
if __name__ == "__main__":
    main()


