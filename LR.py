import numpy as np
import time
#from Data.I_select import Data
#from Data.Select_Best import Data
from Data.Select_From_model import Data
#from Data.All_Features import Data
from sklearn.linear_model import LogisticRegression
from UsefullFunctions import *
from WorkFlow import *
from sklearn.model_selection import LeavePGroupsOut,GroupShuffleSplit

def LR_model(file1):

    # Read Data
    data = Data()
    data_train, data_test, output_train, output_test, unscaled_data_test, activities_labels, feature_labels, groups_at_training, groups_at_testing = data

    InputData(data_train,data_test,output_test,name='Initial Test Data')

    spl = GroupShuffleSplit(n_splits=6, test_size=0.3, random_state=0)
    #spl = GroupShuffleSplit(n_splits=4,train_size=.6, test_size=.4, random_state=42)
    classifier = LogisticRegression(tol=1e-3, penalty='l2')
    classifier_name = LogisticRegression.__name__

    name = "LR"
    params_grid = [
        {'solver': ['newton-cg'], 'C': [0.5, 1, 10, 100, 1000], 'max_iter': [10000, 100000]},
        {'solver': ['lbfgs'], 'C': [0.5, 1, 10, 100, 1000], 'max_iter': [10000, 100000]},
        {'solver': ['liblinear'], 'C': [0.5, 1, 10, 100, 1000], 'max_iter': [10000, 100000]},
        {'solver': ['sag'], 'C': [0.5, 1, 10, 100, 1000], 'max_iter': [10000, 100000]},
        {'solver': ['saga'], 'C': [0.5, 1, 10, 100, 1000], 'max_iter': [10000, 100000]}
    ]

    classifier = CrossValidationWithGridSearch(spl,classifier,data_train,output_train,groups_at_training,params_grid,classifier_name,file1)
    pred_data_test, pred_data_train=model_training(classifier, data_train, output_train, data_test, output_test)

    Results(data_train, output_train, pred_data_train, output_test, pred_data_test, activities_labels, data_test,classifier,file1, name)


if "__main__" == __name__:
    file1 = open("LR.txt", "w")
    time_start = time.time()
    LR_model(file1)
    final_time = time.time() - time_start
    print('Total time for LDA model', final_time)
    file1.writelines('\n Total execution time %d ' % final_time )
    file1.close()