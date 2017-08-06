#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments','bonus','long_term_incentive','expenses','total_stock_value','poi_messages',
				 'exercised_stock_options','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
data_dict.pop('TOTAL',None)
print len(data_dict) # 145 people

#Dictionary storing variables along with the number of their 'NaN' values
nan_dict = {}
for key,value in data_dict.iteritems():
	for k,v in value.iteritems():
		if v == 'NaN':
			if k not in nan_dict:
				nan_dict[k] = 1
			else:
				nan_dict[k] = nan_dict[k] + 1

for key,value in nan_dict.iteritems():
	print key,value

### Task 3: Create new feature(s)
### New feature 'poi_messages' created which is equal to the sum of 'from_poi_to_this_person' and 'from_this_person_to_poi'
for key,value in data_dict.iteritems():
	if value['from_poi_to_this_person'] == 'NaN':
		value['from_poi_to_this_person'] = 0
	if value['from_this_person_to_poi'] == 'NaN':
		value['from_this_person_to_poi'] = 0
	value['poi_messages'] = value['from_poi_to_this_person'] + value['from_this_person_to_poi']
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Alternate classifier used is DecisionTreeClassifier which is named 'dtree' in the pipeline, and is commented so that it can be used
# after commenting GaussianNB classifier.
pipeline = Pipeline([('minmax', MinMaxScaler()),
					 ('skb', SelectKBest()),
					 ('gnb', GaussianNB()),
					# ('dtree', DecisionTreeClassifier())
					])

# DecsionTreeClassifier parameters are commented (to be used when the DecisionTreeClassifier algorithm is used instead of GaussianNB)
param_grid = {	'skb__k': [4,5,6,7,8],
				# 'dtree__criterion': ['gini','entropy'],
				# 'dtree__splitter':['best','random'],
				# 'dtree__min_samples_leaf':[1,2,3,4,5,6],
				# 'dtree__min_samples_split':[2,3,4,5,6,7,8],
				}

gridsearchcv = GridSearchCV(estimator = pipeline, param_grid = param_grid, cv = 4)
gridsearchcv.fit(features_train, labels_train)

print 'Best Parameters from parameter grid'
pprint(gridsearchcv.best_params_)
clf = gridsearchcv.best_estimator_
print clf.named_steps['skb'].scores_
print clf.named_steps['skb'].get_support()
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)