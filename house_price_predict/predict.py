'''
	1. LinearRegression没考虑正则化，交叉验证有大坑，容易某个分组过拟合
	2. 考虑L2正则就是Ridge
	3. 考虑L1正则就是Lasso
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge



class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[self.attribute_names].values


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room=True):
		self.add_bedrooms_per_room = True
		
	def fit(self, x, y=None):
		return self

	def transform(self, x):
		rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
		rooms_per_household = x[:, rooms_ix] / x[:, household_ix]
		population_per_household = x[:, population_ix] / x[:, household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]
			return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
		else:
			return np.c_[x, rooms_per_household, population_per_household]

class MyLabelBinarizer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.label_binarizer = LabelBinarizer()

	def fit(self, x, y=None):
		return self

	def transform(self, x):
		return self.label_binarizer.fit_transform(x)

def read_data(file_path):
	df_data = pd.read_csv(file_path, dtype={"longitude": float, "latitude": float, "housing_median_age": float, "total_rooms": float, "total_bedrooms": float, "population": float, "households": float, "median_income": float, "median_house_value": float, "ocean_proximity": str})
	df_x = df_data.copy().drop("median_house_value", axis=1)
	df_y = df_data[["median_house_value"]].copy()
	return df_x, df_y

def random_split_train_test(x, y, test_ratio=0.2, random_state=10):
	np.random.seed(random_state)
	shuffled_indices = np.random.permutation(len(x))
	test_set_size = int(len(x) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	train_x, train_y, test_x, test_y = x[train_indices, :], y[train_indices, :], x[test_indices, :], y[test_indices, :]
	return (train_x, train_y), (test_x, test_y)

min_max_scaler_x = MinMaxScaler()
min_max_scaler_y = MinMaxScaler()
def preprocess(df_x, df_y):
	attribs = df_x.columns.tolist()
	num_attribs = list(attribs)
	num_attribs.remove("ocean_proximity")
	num_pipeline = Pipeline([
		("selector", DataFrameSelector(num_attribs)),
		("imputer", Imputer(strategy="median", missing_values=np.nan)),
		("combine", CombinedAttributesAdder()),
		("scaler", min_max_scaler_x)
	])
	cat_attribs = ["ocean_proximity"]
	cat_pipeline = Pipeline([
		("selector", DataFrameSelector(cat_attribs)),
		#("labelEncoder", self.label_encoder),
		#("oneHotEncoder", self.one_hot_encoder)
		("label_binarizer", MyLabelBinarizer())
	])
	full_pipeline = FeatureUnion(transformer_list = [
		("num_pipeline", num_pipeline),
		("cat_pipeline", cat_pipeline)
	])
		
	processed_x = full_pipeline.fit_transform(df_x)
	processed_y = min_max_scaler_y.fit_transform(df_y)
	return (processed_x, processed_y)


def evaluate(estimator, x, y):
	y_hat = estimator.predict(x)
	y = min_max_scaler_y.inverse_transform(y)
	y_hat = min_max_scaler_y.inverse_transform(y_hat)
	rmse = np.sqrt(np.sum((y_hat - y) * (y_hat - y)) / len(y))
	return rmse

def score(estimator, x, y):
	y_hat = estimator.predict(x)
	y = min_max_scaler_y.inverse_transform(y)
	y_hat = min_max_scaler_y.inverse_transform(y_hat)
	loss = -1 * np.sum((y_hat - y) * (y_hat - y) / len(y))
	return loss

def cross_validation(estimator, x, y):
	scores = cross_val_score(estimator, x, y, scoring=score, cv=5)
	#scores = cross_val_score(estimator, x, y, scoring="neg_mean_squared_error", cv=10)
	rmse_scores = np.sqrt(-scores)
	print(rmse_scores)
	print("CV Score Mean: %.2f" % (rmse_scores.mean()))
	print("CV Score Std: %.2f" % (rmse_scores.std()))


if __name__ == '__main__':
	df_x, df_y = read_data("housing.csv")
	processed_x, processed_y = preprocess(df_x, df_y)
	(train_x, train_y), (test_x, test_y) = random_split_train_test(processed_x, processed_y)
	ridge = Ridge()
	ridge.fit(train_x, train_y)
	loss = evaluate(ridge, test_x, test_y)
	print("test loss: %.2f" % (loss))
	cross_validation(ridge, processed_x, processed_y)

