import numpy as np
import time
from UsefullFunctions import *
from matplotlib import cm
from sklearn.model_selection import cross_val_score
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib.patches import Patch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import ast

def InputData(data_train,data_test,output_test,name):

    PCA_dunction(data_train, data_test, output_test,name)

def CrossValidationWithoutGridSearch(cv, classifier, X, y, group,alg_name,file1, ax, n_splits, lw=10):

    cmap_data = get_distinct_colors(30)
    cmap_data_class = plt.cm.Paired
    cmap_cv = plt.cm.coolwarm

    acur_test = []
    time_start = time.time()
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        X_train, X_test = X[tr], X[tt]
        y_train, y_test = y[tr], y[tt]
        clf = classifier
        final_model = clf.fit(X_train, y_train)
        pred_test = final_model.predict(X_test)
        acur_test.append(accuracy_score(y_test, pred_test))

        print(acur_test)
        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    final_time = time.time() - time_start

    file1.writelines('Time for Cross Validation %f' % final_time)
    file1.writelines("\nMean accuracy of the sample is % s " % (statistics.mean(acur_test)))
    file1.writelines("\nStandard Deviation of accuracy at the sample is % s " % (statistics.stdev(acur_test)))

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=sorted(y), marker='_', lw=lw, cmap=cmap_data_class)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 2.2, -.2], xlim=[0, 8000])

    ax.set_title('{0}: {1}'.format(alg_name,(type(cv).__name__)), fontsize=15)

    ax.legend([Patch(color=cmap_cv(.8)), Patch(color=cmap_cv(.02))],
              ['Testing set', 'Training set'], loc=(1.02, .8))
    plt.tight_layout()
    plt.show()

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(classifier, X, y, groups=group, cv=cv,
                       train_sizes=np.linspace(.1, 1.0, 5),
                       return_times=True)
    plot_learning_curve("LDA Learning rate curve", train_sizes, train_scores, test_scores, fit_times)
    plt.show()

def CrossValidationWithGridSearch(cv,classifier,data_train,output_train,groups_at_training,params_grid,classifier_name,file1):

    time_start = time.time()
    print(type(cv))
    #classifier = ast.literal_eval(classifier)
    print(type(classifier))
    print(classifier)

    GS_model = GridSearchCV(classifier, params_grid, cv=cv)
    final_model = GS_model.fit(data_train, output_train, groups=groups_at_training)


    final_time = time.time() - time_start
    file1.writelines('\nTime for Cross Validation %d' % final_time)

    if classifier_name == 'LogisticRegression':

    # View the accuracy score

        file1.writelines('\nBest accuracy score for training data: %f'% GS_model.best_score_)
        # View the best parameters for the model found using grid search

        file1.writelines('\nBest C: %f' % GS_model.best_estimator_.C)
        file1.writelines('\nBest Solver: %s' % GS_model.best_estimator_.solver)
        file1.writelines('\nBest max_iter:: %s' % GS_model.best_estimator_.max_iter)

        s = str(GS_model.best_estimator_.solver)
        c = GS_model.best_estimator_.C
        m_i = GS_model.best_estimator_.max_iter

        classifier = LogisticRegression(solver=s, C=c, max_iter=m_i, tol=1e-3)

    elif classifier_name == 'SVC':

        file1.writelines('\nBest accuracy score for training data: %f' % GS_model.best_score_)
        # View the best parameters for the model found using grid search

        file1.writelines('\nBest C: %f' % GS_model.best_estimator_.C)
        file1.writelines('\nBest Kernel: %s' % GS_model.best_estimator_.kernel)
        file1.writelines('\nBest Gamma: %s' % GS_model.best_estimator_.gamma)
        file1.writelines('\nBest Degree: %s' % GS_model.best_estimator_.degree)

        k = str(GS_model.best_estimator_.kernel)
        c = GS_model.best_estimator_.C
        d = GS_model.best_estimator_.degree
        g = GS_model.best_estimator_.gamma
        classifier = SVC(C=c, kernel=k, degree=d, gamma=g)

        print('classifier', classifier)

    elif classifier_name == 'DecisionTreeClassifier':

        file1.writelines('\nBest Score: %s' % GS_model.best_score_)
        file1.writelines('\nBest C: %s' % GS_model.best_estimator_.criterion)
        file1.writelines('\nMax leaf nodes: %s' % GS_model.best_estimator_.max_leaf_nodes)


        crit = str(GS_model.best_estimator_.criterion)
        max_leaf_nodes_ = GS_model.best_estimator_.max_leaf_nodes


        classifier = DecisionTreeClassifier(criterion=crit, max_leaf_nodes=max_leaf_nodes_)
        print('classifier', classifier)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(classifier, data_train, output_train, groups=groups_at_training, cv=cv,
                       train_sizes=np.linspace(.1, 1.0, 5),
                       return_times=True)
    plot_learning_curve("LR Learning rate curve", train_sizes, train_scores, test_scores, fit_times)
    plt.show()
    return classifier

def model_training(classifier,data_train,output_train,data_test,output_test,group_training):

    print(classifier)
    final_model = classifier.fit(data_train, output_train,group_training)
    pred_data_test = final_model.predict(data_test)
    pred_data_train = final_model.predict(data_train)

    print(accuracy_score(output_test, pred_data_test))
    print(accuracy_score(output_train,pred_data_train))
    return pred_data_test, pred_data_train

def Results(data_train, output_train,pred_train,output_test ,pred_test,activities_labels,data_test,clf,file1,name):

    file1.writelines('\nScores_for_test %s' % classification_report(output_test, pred_test, target_names=activities_labels))
    file1.writelines('\nScores_for_train %s' % classification_report(output_train, pred_train, target_names=activities_labels))

    cm_test = confusion_matrix(output_test, pred_test, normalize='true')
    cm_train = confusion_matrix(output_train, pred_train, normalize='true')
    plot_confusion_matrix(cm_test, activities_labels,name+": Data from Testing Set")
    plot_confusion_matrix(cm_train, activities_labels, name+": Data from Training Set")

    PCA_dunction(data_train, data_test, pred_test, "Predicted Data")
    Test_Pred_plots(data_test, output_test, pred_test,name)

    ROC_Curve(data_train, data_test, output_train, output_test, clf,name)

