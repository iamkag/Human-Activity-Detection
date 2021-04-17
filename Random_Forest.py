from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from Data.Select_From_model import Data
from WorkFlow import *
from sklearn.model_selection import LeavePGroupsOut,GroupShuffleSplit


def RandomForest(file1):
    # Read Data
    data = Data()
    data_train, data_test, output_train, output_test, unscaled_data_test, activities_labels, feature_labels, groups_at_training, groups_at_testing = data

    InputData(data_train, data_test, output_test, name='Initial Test Data')

    spl = GroupShuffleSplit(n_splits=6, test_size=0.3, random_state=0)
    classifier = RandomForestClassifier(n_estimators=100, criterion='gini',max_depth=70, random_state=50)
    fig, ax = plt.subplots()
    name = "Random Forest"
    alg_name = RandomForestClassifier.__name__
    CrossValidationWithoutGridSearch(spl, classifier, data_train, output_train, groups_at_training, alg_name, file1, ax, n_splits=6, lw=10)

    pred_data_test, pred_data_train = model_training(classifier, data_train, output_train, data_test, output_test)

    Results(data_train, output_train, pred_data_train, output_test, pred_data_test, activities_labels, data_test,
            classifier, file1, name)

if "__main__" == __name__:
    file1 = open("RandomForest.txt", "w")
    time_start = time.time()
    RandomForest(file1)
    final_time = time.time() - time_start
    print('Total time for LDA model', final_time)
    file1.writelines('\n Total execution time %d ' % final_time)
    file1.close()