# Compare Algorithms
import pandas
import numpy as np
import matplotlib.pyplot as plt
from Data.Select_From_model import Data
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from UsefullFunctions import Add_Noise
from sklearn.preprocessing import MinMaxScaler
data = Data()
data_train, data_test, output_train, output_test, unscaled_data_test, activities_labels, feature_labels, groups_at_training, groups_at_testing = data



# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression(C=10,solver ='liblinear',max_iter= 10000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('RF', RandomForestClassifier(n_estimators=1000, criterion='gini',max_depth=20, random_state=50)))
models.append(('DTree', DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=70)))
models.append(('SVC', SVC(C=100, kernel='rbf', degree=3, gamma=0.001)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
	cv_results=[]
	sfs = model_selection.GroupShuffleSplit(n_splits=6, test_size=0.3, random_state=0)
	for tr, tt in sfs.split(data_train, output_train, groups_at_training):
		# Fill in indices with the training/test groups
		X_train, X_test = data_train[tr], data_train[tt]
		y_train, y_test = output_train[tr], output_train[tt]
		if name=='LDA':
			final_model = model.fit(X_train, y_train)
		else:
			final_model = model.fit(X_train, y_train, groups_at_training[tr])

		pred_test = final_model.predict(X_test)
		cv_results.append(accuracy_score(y_test, pred_test))
		print(cv_results)

	names.append(name)
	results.append(np.array(cv_results))
	msg = "%s: %f (%f)" % (name, np.array(cv_results).mean(), np.array(cv_results).std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison at Test Set')
ax = fig.add_subplot(111)
print(results)
print(type(results))
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()