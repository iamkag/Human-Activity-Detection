import numpy as np
import time

from Data.Select_From_model import Data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from UsefullFunctions import *
from WorkFlow import *
from sklearn.model_selection import LeavePGroupsOut,GroupShuffleSplit

def LDA_model(file1):

    # Read Data
    data = Data()
    data_train,data_test,output_train,output_test,unscaled_data_test,activities_labels,feature_labels,groups_at_training,groups_at_testing = data
    num_training = cound_groups(groups_at_training)
    num_test = cound_groups(groups_at_testing)
    print(num_training,num_test)

    PlotTheInputData(output_train, activities_labels, "Train Data")
    PlotTheInputData(output_test, activities_labels, 'Test Data')

    InputData(data_train,data_test,output_test,name='Initial Test Data')

    #lpl = LeavePGroupsOut(n_groups=2)
    spl = GroupShuffleSplit(n_splits=6, test_size=0.3, random_state=0)
    classifier = LinearDiscriminantAnalysis()
    fig, ax = plt.subplots()
    name = "LDA"
    alg_name = LinearDiscriminantAnalysis.__name__
    CrossValidationWithoutGridSearch(spl, classifier, data_train, output_train, groups_at_training,alg_name,file1, ax, n_splits=6, lw=10)

    pred_data_test, pred_data_train=model_training(classifier, data_train, output_train, data_test, output_test)

    Results(data_train, output_train, pred_data_train, output_test, pred_data_test, activities_labels, data_test,classifier,file1, name)


if "__main__" == __name__:
    file1 = open("LDA.txt", "w")
    time_start = time.time()

    LDA_model(file1)
    final_time = time.time() - time_start
    print('Total time for LDA model', final_time)
    file1.writelines('\n Total execution time %d ' % final_time )
    file1.close()

