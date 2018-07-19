
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion




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

class HousePricePredict(object):
	def __init__(self):
		self.imputer = Imputer(strategy="median", missing_values=np.nan)
		self.min_max_scaler_x = MinMaxScaler()
		self.min_max_scaler_y = MinMaxScaler()
		self.label_binarizer = MyLabelBinarizer()

	def read_data(self, file_path):
		self.data = pd.read_csv(file_path, dtype={"longitude": float, "latitude": float, "housing_median_age": float, "total_rooms": float, "total_bedrooms": float, "population": float, "households": float, "median_income": float, "median_house_value": float, "ocean_proximity": str})
		self.x = self.data.copy().drop("median_house_value", axis=1)
		self.y = self.data[["median_house_value"]].copy()

	def random_split_train_test(self, test_ratio=0.2, random_state=42):
		np.random.seed(random_state)
		shuffled_indices = np.random.permutation(len(self.data))
		test_set_size = int(len(self.data) * test_ratio)
		test_indices = shuffled_indices[:test_set_size]
		train_indices = shuffled_indices[test_set_size:]
		self.train_x, self.train_y = self.processed_x[train_indices, :], self.processed_y[train_indices, :]
		self.test_x, self.test_y = self.processed_x[test_indices, :], self.processed_y[test_indices, :]


	def preprocessing(self):
		attribs = self.x.columns.tolist()
		num_attribs = list(attribs)
		num_attribs.remove("ocean_proximity")
		cat_attribs = ["ocean_proximity"]
		num_pipeline = Pipeline([
			("selector", DataFrameSelector(num_attribs)),
			("imputer", self.imputer),
			("combine", CombinedAttributesAdder()),
			("scaler", self.min_max_scaler_x)
			])
		cat_pipeline = Pipeline([
			("selector", DataFrameSelector(cat_attribs)),
			#("labelEncoder", self.label_encoder),
			#("oneHotEncoder", self.one_hot_encoder)
			("label_binarizer", self.label_binarizer)
			])
		full_pipeline = FeatureUnion(transformer_list = [
			("num_pipeline", num_pipeline),
			("cat_pipeline", cat_pipeline)
			])
		
		self.processed_x = full_pipeline.fit_transform(self.x)
		self.processed_y = self.min_max_scaler_y.fit_transform(self.y)
		print('processed_x:')
		print(self.processed_x[0:10, ])
		print('processed_y:')
		print(self.processed_y[0:10, ])

	def model(self):
		from sklearn.linear_model import LinearRegression
		self.lr = LinearRegression()
		self.lr.fit(self.train_x, self.train_y)

	def predict(self, x):
		y = self.lr.predict(x)
		return self.min_max_scaler_y.inverse_transform(y)

	def evalute(self, x, y):
		y_hat = self.predict(x)
		y = self.min_max_scaler_y.inverse_transform(y)
		print(y_hat[0:10, ])
		print(y[0:10, ])
		loss = np.sqrt(np.sum((y_hat - y) * (y_hat - y)) / len(y))
		return loss
predict = HousePricePredict()
predict.read_data("housing.csv")
predict.preprocessing()
predict.random_split_train_test()
predict.model()
loss = predict.evalute(predict.test_x, predict.test_y)
print("test loss: %.2f" % (loss))


