import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import itertools
from IPython.display import display
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from scipy import interp
from itertools import cycle
from sklearn.metrics import auc
from colorsys import hls_to_rgb
import matplotlib

#Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
#The input data is centered but not scaled for each feature before applying the SVD.
def PCA_dunction(X_train,X_test,y_test,title):
    pca2 = PCA(n_components=2)
    pca2.fit(X_train)
    X_train_2 = pca2.transform(X_train)
    X_test_2 = pca2.transform(X_test)


    x11 = []
    x12 = []
    x21 = []
    x22 = []
    x31 = []
    x32 = []
    x41 = []
    x42 = []
    x51 = []
    x52 = []
    x61 = []
    x62 = []

    for i in range(len(y_test)):
        if (y_test[i] == 1):
            x11.append(X_test_2[i][0])
            x12.append(X_test_2[i][1])
        elif (y_test[i] == 2):
            x21.append(X_test_2[i][0])
            x22.append(X_test_2[i][1])
        elif (y_test[i] == 3):
            x31.append(X_test_2[i][0])
            x32.append(X_test_2[i][1])
        elif (y_test[i] == 4):
            x41.append(X_test_2[i][0])
            x42.append(X_test_2[i][1])
        elif (y_test[i] == 5):
            x51.append(X_test_2[i][0])
            x52.append(X_test_2[i][1])
        else:
            x61.append(X_test_2[i][0])
            x62.append(X_test_2[i][1])

    plt.figure()
    plt.plot(x41, x42, 'xr', label='4-Sitting')
    plt.plot(x51, x52, 'xm', label='5-Standing')
    plt.plot(x11, x12, 'xc', label='1-Walking')
    plt.plot(x21, x22, 'xb', label='2-Upstairs')
    plt.plot(x31, x32, 'xy', label='3-Downstairs')
    plt.plot(x61, x62, 'xg', label='6-Laying')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.show()

def plot_confusion_matrix(cm, classes,matrix_title,
                          normalize=True,
                          cmap=plt.cm.Blues):
    """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')



    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(matrix_title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def PlotTheInputData(output_train, activities_labels,plot_name):

    # Visualizing Outcome Distribution
    temp = pd.Index(output_train).value_counts()
    df = pd.DataFrame({'labels': temp.index,
                       'values': temp.values
                       })

    # df.plot(kind='pie',labels='labels',values='values', title='Activity Ditribution',subplots= "True")
    labels = activities_labels
    sizes = df['values']
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'cyan', 'lightpink']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.title(plot_name)
    plt.show()

def GridSearch_table_plot(grid_clf, param_name,
                          num_results=15,
                          negative=True,
                          graph=True,
                          display_all_params=True):

    '''Display grid search results

    Arguments
    ---------

    grid_clf           the estimator resulting from a grid search
                       for example: grid_clf = GridSearchCV( ...

    param_name         a string with the name of the parameter being tested

    num_results        an integer indicating the number of results to display
                       Default: 15

    negative           boolean: should the sign of the score be reversed?
                       scoring = 'neg_log_loss', for instance
                       Default: True

    graph              boolean: should a graph be produced?
                       non-numeric parameters (True/False, None) don't graph well
                       Default: True

    display_all_params boolean: should we print out all of the parameters, not just the ones searched for?
                       Default: True

    Usage
    -----

    GridSearch_table_plot(grid_clf, "min_samples_leaf")

                          '''

    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if display_all_params:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + param_name]

    # display the top 'num_results' results
    # =====================================
    display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    # plot the results
    # ================
    scores_df = scores_df.sort_values(by='param_' + param_name)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + param_name]

    # plot
    if graph:
        plt.figure(figsize=(8, 8))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.show()

def Plot_Validation_Curves(train_scores,test_scores,param_range,name):

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel(name)
    plt.ylabel("Score")
    plt.ylim(0.5, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()



def plot_learning_curve(title,train_sizes,train_scores,test_scores,fit_times,axes=None, ylim=None):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("accuracy")

    #train_sizes, train_scores, test_scores, fit_times, _ = \
    #    learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
    #                   train_sizes=train_sizes,
    #                   return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training accuracy")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation accuracy")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("Performance of the model")

    return plt

def Va_Test(model,X_train,y_train,X_test,y_test,name_of_optim):

   #np.random.seed(1)
   #""" Example based on sklearn's docs """
   #mnist = fetch_openml('mnist_784')
   ## rescale the data, use the traditional train/test split
   #X, y = mnist.data / 255., mnist.target
   #X_train, X_test = X[:60000], X[60000:]
   #y_train, y_test = y[:60000], y[60000:]

    #mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
    #                    solver='adam', verbose=0, tol=1e-8, random_state=1,
    #                    learning_rate_init=.01)

    """ Home-made mini-batch learning
        -> not to be used in out-of-core setting!
    """
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 100
    N_BATCH = 256
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []
    y_train = np.asarray(y_train)

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]

            model.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        scores_train.append(model.score(X_train, y_train))

        # SCORE TEST
        scores_test.append(model.score(X_test, y_test))

        epoch += 1

    """ Plot """
    #fig, ax = plt.subplots(2, sharex=True, sharey=True)
    #ax[0].plot(scores_train)
    #ax[0].set_title('Train')
    #ax[1].plot(scores_test)
    #ax[1].set_title('Test')
    #fig.suptitle("Accuracy over epochs", fontsize=14)
    plt.plot(scores_train, label='Train')
    plt.plot(scores_test,label='Test')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.title(name_of_optim)
    plt.legend()
    plt.show()


def Test_Pred_plots(unscaled_data_test,output_test,y_pred_test,title):
    data_test_one_col = []
    for i in range(len(unscaled_data_test)):
        data_test_one_col.append(unscaled_data_test[i][0])
    #
    #print(max(data_test_one_col))
    #print(min(data_test_one_col))

    plt.scatter(data_test_one_col, output_test, color='red', linewidths=2)
    plt.scatter(data_test_one_col, y_pred_test, color='green', linewidths=0.05)
    #
    plt.xticks((np.arange(-1.5, 0.7, step=200)))
    plt.yticks((np.arange(0, 8, step=1)))
    plt.title(title)
    plt.xlabel('dx')
    plt.ylabel('Activity type')
    plt.legend(("Actual", "Predicted"))
    plt.show()

def ROC_Curve(data_train,data_test,output_train,output_test,model,name):

    # Binarize the output
    output_train = label_binarize(output_train, classes=[1, 2, 3, 4, 5, 6])
    n_classes = output_train.shape[1]


    output_test = label_binarize(output_test, classes=[1, 2, 3, 4, 5, 6])
    n_classes = output_test.shape[1]


    classifier = OneVsRestClassifier(model)

    y_score = classifier.fit(data_train, output_train).decision_function(data_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(output_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(output_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this pointsΦλεαρ
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=i,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=i)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name+':ROC for test set')
    plt.legend(loc="lower right")

    plt.show()



def error(y_test, y_predict, name):
    ercount = 0

    for p in range(len(y_test)):
        if (y_test[p] != y_predict[p]):
            ercount += 1

    test_rate = ercount * 1.0 / len(y_test)

    print("The test error rate of", name, "is", test_rate)

def get_distinct_colors(n):

    cmaps =['gist_rainbow', 'nipy_spectral', 'gist_ncar']
    segmented_cmaps = matplotlib.colors.ListedColormap(plt.get_cmap('gist_rainbow')(np.linspace(0,1,n)))

    return segmented_cmaps

def Add_Noise(data_train,data_test,output_test,name1,name2):
    PCA_dunction(data_train, data_test, output_test, name1)
    n_samples_train, n_features_train = data_train.shape
    n_samples_test, n_features_test = data_test.shape
    random_state = np.random.RandomState(0)
    data_train = np.c_[data_train, random_state.randn(n_samples_train, 2 * n_features_train)]
    data_test = np.c_[data_test, random_state.randn(n_samples_test, 2 * n_features_test)]
    PCA_dunction(data_train, data_test, output_test,name2)
    print("Data_train", np.shape(data_train), "Data_test", np.shape(data_test))
    return data_train,data_test

def cound_groups(groups):
    num=1
    x=groups[0]
    for el in groups:
        if el != x:
            num+=1
            x=el
    return num

