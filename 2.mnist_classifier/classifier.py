#coding: utf8
import pandas as pd
import scipy.io
import gzip
import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def _read32(bytestream):
	dt = np.dtype(np.uint32).newbyteorder('>')
	return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
	print('Extracting', f.name)
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = _read32(bytestream)
		if magic != 2051:
			raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, f.name))
		num_images = _read32(bytestream)
		rows = _read32(bytestream)
		cols = _read32(bytestream)
		buf = bytestream.read(rows * cols * num_images)
		data = np.frombuffer(buf, dtype=np.uint8)
		data = data.reshape(num_images, rows * cols)
	return data

def extract_labels(f):
	print('Extracting', f.name)
	with gzip.GzipFile(fileobj=f) as bytestream:
		magic = _read32(bytestream)
		if magic != 2049:
			raise ValueError('Invalid magic number %d in MNIST label file: %s' %(magic, f.name))
		num_items = _read32(bytestream)
		buf = bytestream.read(num_items)
		labels = np.frombuffer(buf, dtype=np.uint8)
	return labels

def read_data():
	train_images = 'train-images-idx3-ubyte.gz'
	train_labels = 'train-labels-idx1-ubyte.gz'
	test_images = 't10k-images-idx3-ubyte.gz'
	test_labels = 't10k-labels-idx1-ubyte.gz'

	local_dir = './mldata/'
	local_file = local_dir + train_images
	with open(local_file, 'rb') as f:
		train_x = extract_images(f)

	local_file = local_dir + test_images
	with open(local_file, 'rb') as f:
		test_x = extract_images(f)

	local_file = local_dir + train_labels
	with open(local_file, 'rb') as f:
		train_y = extract_labels(f)

	local_file = local_dir + test_labels
	with open(local_file, 'rb') as f:
		test_y = extract_labels(f)

	mnist = {}
	#mnist['data'] = np.r_[train_x, test_x]
	#mnist['target'] = np.r_[train_y, test_y]
	mnist['train_x'] = train_x
	mnist['train_y'] = train_y
	mnist['test_x'] = test_x
	mnist['test_y'] = test_y
	return mnist

mnist = read_data()
train_x, train_y, test_x, test_y = mnist['train_x'], mnist['train_y'], mnist['test_x'], mnist['test_y']
#plot sample digit 9
some_digit = mnist['train_x'][36000]
some_digit_label = mnist['train_y'][36000]
some_digit_image = some_digit.reshape(28, 28)
print('some digit label:', some_digit_label)
'''
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()
'''

'''
	train an binary classifier
'''
train_y_9 = train_y == 9
test_y_9 = test_y == 9
sgd_clf = SGDClassifier(random_state=42)
#sgd_clf.fit(train_x, train_y_9)
sgd_clf.fit(train_x, train_y)
print(sgd_clf.predict([some_digit]))

'''
	performance measure
def customized_cross_validation(x, y):
	skfolds = StratifiedKFold(n_splits=3, random_state=42)
	scores = []
	for train_index, test_index in skfolds.split(x, y):
		clone_sgd_clf = clone(sgd_clf)
		train_x = x[train_index]
		train_y = y[train_index]
		test_x = x[test_index]
		test_y = y[test_index]
		clone_sgd_clf.fit(train_x, train_y)
		y_pred = clone_sgd_clf.predict(test_x)
		n_correct = sum(y_pred == test_y)
		accuracy = n_correct / len(y_pred)
		scores.append(accuracy)
	return scores

cu_cv = customized_cross_validation(train_x, train_y_9)
sk_cv = cross_val_score(sgd_clf, train_x, train_y_9, cv=3, scoring="accuracy")
print('Customized CV: precision', cu_cv)
print('Sklearn CV: precision', sk_cv)
train_pred = cross_val_predict(sgd_clf, train_x, train_y_9, cv=3)
cm = confusion_matrix(train_y_9, train_pred)
print('precision:', precision_score(train_y_9, train_pred))
print('recall:', recall_score(train_y_9, train_pred))
train_score_pred = cross_val_predict(sgd_clf, train_x, train_y_9, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(train_y_9, train_score_pred)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
	plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
	plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
	plt.xlabel("Threshold")
	plt.legend(loc="upper left")
	plt.ylim([0, 1])
	plt.show()
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
print('Roc-Auc Score:', roc_auc_score(train_y_9, train_score_pred))
'''

def plot_roc_curve(fpr, tpr, label=None):
	plt.plot(fpr, tpr, linewidth=2, label=label)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	
def plot_precision_recall(precision, recall, label=None):
	plt.plot(recall, precision, linewidth=2, label=label)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([0, 1, 0, 1])
	plt.xlabel('recall')
	plt.ylabel('precision')

def compare_roc_sgd_vs_random_forest(x, y):
	sgd_clf = SGDClassifier(random_state=42)
	forest_clf = RandomForestClassifier(random_state=42)
	y_sgd = cross_val_predict(sgd_clf, x, y, cv=3, method='decision_function')
	y_probas_forest = cross_val_predict(forest_clf, x, y, cv=3, method='predict_proba')
	y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
	fpr_forest, tpr_forest, thresholds_forest = roc_curve(y, y_scores_forest)
	fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y, y_sgd)
	plt.plot(fpr_sgd, tpr_sgd, 'b:', label='SGD')
	plot_roc_curve(fpr_forest, tpr_forest, label='RanodmForest')
	plt.legend(loc='lower right')
	plt.show()

def compare_pr_sgd_vs_random_forest(x, y):
	sgd_clf = SGDClassifier(random_state=42)
	forest_clf = RandomForestClassifier(random_state=42)
	y_sgd = cross_val_predict(sgd_clf, x, y, cv=3, method='decision_function')
	y_probas_forest = cross_val_predict(forest_clf, x, y, cv=3, method='predict_proba')
	y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
	precision_forest, recall_forest, thresholds_forest = precision_recall_curve(y, y_scores_forest)
	precision_sgd, recall_sgd, thresholds_sgd = precision_recall_curve(y, y_sgd)
	print(precision_sgd.shape)
	print(recall_sgd.shape)
	print(precision_forest.shape)
	print(recall_forest.shape)
	np.savetxt('precision_sgd.csv', precision_sgd, delimiter=',')
	np.savetxt('precision_forest.csv', precision_forest, delimiter=',')
	np.savetxt('recall_sgd.csv', recall_sgd, delimiter=',')
	np.savetxt('recall_forest.csv', recall_forest, delimiter=',')
	plt.plot(precision_sgd, recall_sgd, 'b:', label='SGD')
	plot_precision_recall(precision_forest, recall_forest, label='RanodmForest')
	plt.legend(loc='lower right')
	plt.show()
#compare_roc_sgd_vs_random_forest(train_x, train_y_9)
#compare_pr_sgd_vs_random_forest(train_x, train_y_9)
