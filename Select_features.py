# Compare Algorithms
import pandas
import numpy as np
import matplotlib.pyplot as plt
from Data.Select_From_model import Data
from Data.RandomForest import RF
from Data.LassoCV import LassoCV_
from Data.Select_Best import Best125
from Data.Select_Best193 import Best193
from Data.I_select import Iselect
from sklearn import model_selection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import accuracy_score

data = Data()
data_linear_SVC= data[0]
output_train = data[2]
groups_at_training = data[7]
data_rf= RF()[0]
data_Lasso= LassoCV_()[0]
data_125 = Best125()[0]
data_193 = Best193()[0]
data_select = Iselect()[0]
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('Lasso', LinearDiscriminantAnalysis(),data_Lasso))
models.append(('RanFor', LinearDiscriminantAnalysis(),data_rf))
models.append(('LSVC', LinearDiscriminantAnalysis(),data_linear_SVC))
models.append(('Best_125', LinearDiscriminantAnalysis(),data_125))
models.append(('Best_193', LinearDiscriminantAnalysis(),data_193))
models.append(('I_select', LinearDiscriminantAnalysis(),data_select))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model,data_ in models:
	cv_results=[]
	sfs = model_selection.GroupShuffleSplit(n_splits=6, test_size=0.3, random_state=0)
	for tr, tt in sfs.split(data_, output_train, groups_at_training):
		# Fill in indices with the training/test groups
		X_train, X_test = data_[tr], data_[tt]
		y_train, y_test = output_train[tr], output_train[tt]
		final_model = model.fit(X_train, y_train)
		pred_test = final_model.predict(X_test)
		cv_results.append(accuracy_score(y_test, pred_test))
		print(cv_results)

	names.append(name)
	results.append(np.array(cv_results))
	msg = "%s: %f (%f)" % (name, np.array(cv_results).mean(), np.array(cv_results).std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Accuracy VS Model for feature selection')
ax = fig.add_subplot(111)
print(results)
print(type(results))
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()