import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def Data():
    activities_labels = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/activity_labels.csv", header=None)
    activities_labels = pd.DataFrame(activities_labels).to_numpy()
    activities_labels = list(activities_labels.flatten())

    feature_labels = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/features.csv", header=None, delim_whitespace=True)

    groups_at_training = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/train/subject_train.txt", header=None)
    groups_at_training = pd.DataFrame(groups_at_training).to_numpy()
    groups_at_training = groups_at_training.reshape(len(groups_at_training), )
    # groups_at_training = list(groups_at_training.flatten())

    groups_at_testing = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/test/subject_test.txt", header=None)
    groups_at_testing = pd.DataFrame(groups_at_testing).to_numpy()
    groups_at_testing = groups_at_testing.reshape(len(groups_at_testing), )
    # groups_at_testing = list(groups_at_testing.flatten())

    # Specify data
    data_train = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/train/X_train.csv", delim_whitespace=True,
                             header=None)
    data_test = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/test/X_test.csv", delim_whitespace=True, header=None)
    data_train = pd.DataFrame(data_train).to_numpy()
    data_test = pd.DataFrame(data_test).to_numpy()

    output_train = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/train/y_train.csv", delim_whitespace=True,
                               header=None)
    output_test = pd.read_csv("/Users/user/Desktop/UCI_HAR_Dataset/test/y_test.csv", delim_whitespace=True, header=None)
    output_train = pd.DataFrame(output_train).to_numpy()
    output_test = pd.DataFrame(output_test).to_numpy()
    output_train = output_train.reshape(len(groups_at_training), )
    output_test = output_test.reshape(len(output_test), )
    # output_train = list(output_train.flatten())
    # output_test = list(output_test.flatten())


    unscaled_data_test = data_test

    print("Data_train", np.shape(data_train), "Data_test", np.shape(data_test), "Output_train", np.shape(output_train),
          "Output_test", np.shape(output_test), "activities_labels", np.shape(activities_labels), "groups_at_training",
          np.shape(groups_at_training), "groups_at_testing", np.shape(groups_at_testing))

    print("################################")

    return data_train, data_test, output_train, output_test, unscaled_data_test, activities_labels,feature_labels, groups_at_training , groups_at_testing

if "__main__" == __name__:
    Data()