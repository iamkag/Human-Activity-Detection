import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from Data.Select_From_model import Data
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import sklearn.metrics as metrics
from sklearn import tree
from UsefullFunctions import *
from WorkFlow import *
from sklearn.model_selection import LeavePGroupsOut,GroupShuffleSplit


def Plot_Decision_Tree(clf):

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i]))

    plt.figure(figsize=(90,50))
    tree.plot_tree(clf,fontsize=5)
    plt.show()


def DTree(file1):
    # Read Data
    data = Data()
    data_train, data_test, output_train, output_test, unscaled_data_test, activities_labels, feature_labels, groups_at_training, groups_at_testing = data

    InputData(data_train, data_test, output_test, name='Initial Test Data')

    spl = GroupShuffleSplit(n_splits=6, test_size=0.3, random_state=0)
    classifier = DecisionTreeClassifier()
    classifier_name = DecisionTreeClassifier.__name__
    name = "DTree"
    params_grid = [{'criterion':["gini","entropy"],'max_leaf_nodes':[30,50,70,100]}]

    classifier = CrossValidationWithGridSearch(spl, classifier, data_train, output_train, groups_at_training,
                                                params_grid, classifier_name, file1)
    pred_data_test, pred_data_train = model_training(classifier, data_train, output_train, data_test, output_test)

    Results(data_train, output_train, pred_data_train, output_test, pred_data_test, activities_labels, data_test,
            classifier, file1, name)

if "__main__" == __name__:
    file1 = open("DTree.txt", "w")
    time_start = time.time()
    DTree(file1)
    final_time = time.time() - time_start
    print('Total time for LDA model', final_time)
    file1.writelines('\n Total execution time %d ' % final_time)
    file1.close()