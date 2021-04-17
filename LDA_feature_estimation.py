import numpy as np
import time
#from Data.I_select import Data
#from Data.Select_Best import Data
#from Data.Select_From_model import Data
from Data.All_Features import Data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from UsefullFunctions import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeavePGroupsOut,GroupShuffleSplit
from matplotlib import cm
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
import statistics

def cound_groups(groups):
    num=1
    x=groups[0]
    for el in groups:
        if el != x:
            num+=1
            x=el
    return num

def LDA_model():

    # Read Data
    data = Data()
    data_train = data[0]
    data_test = data[1]
    output_train = data[2]
    output_test = data[3]
    unscaled_data_test = data[4]
    activities_labels = data[5]
    feature_labels = data[6]
    groups_at_training = data[7]
    groups_at_testing = data[8]

    num_training = cound_groups(groups_at_training)
    num_test = cound_groups(groups_at_testing)

    LDA = LinearDiscriminantAnalysis()
   #slp = LeavePGroupsOut(n_groups=2)
    slp = GroupShuffleSplit(n_splits=2, train_size=.3, random_state=42)

    min_features_to_select = 1
    rfecv = RFECV(estimator=LDA, step=1, cv=slp,
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select)
    rfecv.fit(data_train, output_train, groups_at_training)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)
    plt.show()




if "__main__" == __name__:
    LDA_model()


