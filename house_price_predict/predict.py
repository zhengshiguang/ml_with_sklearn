import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import cm as cm

random_state = 42

def read_data():
	data = pd.read_csv('housing.csv', dtype={'longitude': float, 'latitude': float, 'housing_median_age': float, "total_rooms": float, "total_bedrooms": float, "population": float, "households": float, 'median_income': float, 'median_house_value': float, 'ocean_proximity': str})

	median = data['total_bedrooms'].median()
	data['total_bedrooms'] = data['total_bedrooms'].fillna(median)
	data['income_cat'] = np.ceil(data['median_income'] / 1.5)
	data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
	return data

def random_split_train_test(data, test_ratio):
	'''
		set random seed, keep train&test same overtime
	'''
	np.random.seed(random_state)
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

def stratified_split_train_test(data, test_ratio):
	
	split = StratifiedShuffleSplit(n_splits=1, test_size = test_ratio, random_state=random_state)
	for train_indices, test_indices in split.split(data, data['income_cat']):
		strat_train_set = data.iloc[train_indices]
		strat_test_set = data.iloc[test_indices]
	return strat_train_set, strat_test_set

def describe(df_house):
	'''
	print("info:")
	print(df_house.info())
	
	print("describe:")
	print(df_house.describe())
	'''
	df_house.hist(bins=50, figsize=(28, 15))
	plt.show()

def compare_random_split_and_stratified_split(data):
	random_train, random_test = random_split_train_test(data, 0.2)
	strat_train, strat_test = stratified_split_train_test(data, 0.2)

	df = pd.DataFrame()
	df['overall'] = data['income_cat'].value_counts() / len(data)
	df['random_train'] = random_train['income_cat'].value_counts() / len(random_train)
	df['random_test'] = random_test['income_cat'].value_counts() / len(random_test)
	df['strat_train'] = strat_train['income_cat'].value_counts() / len(strat_train)
	df['strat_test'] = strat_test['income_cat'].value_counts() / len(strat_test)
	print("random split vs. stratified split:")
	print(df.sort_index())

def visulation_and_discover(data):
	copy_data = data.copy()
	del copy_data['income_cat']
	'''
	# plot corr matrix
	print("feature corr matrix:")
	corr = copy_data.corr()
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.grid(True)
	ax1.set_title('House Price Feature Correlation')
	labels = corr.columns.tolist()
	ax1.set_xticks(range(0, len(labels)))
	ax1.set_yticks(range(0, len(labels)))
	ax1.set_xticklabels(labels, fontsize=8)
	ax1.set_yticklabels(labels, fontsize=8)
	cmap = cm.get_cmap('jet', 30)
	cax = ax1.imshow(corr.values, interpolation="nearest", cmap=cmap)
	fig.colorbar(cax, ticks=np.arange(-1, 1, 0.1))
	plt.savefig('xx.png')
	plt.show()
	'''
	
	'''
	# housing price
	copy_data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=copy_data['population']/100, c=copy_data['median_house_value']/100, cmap=cm.get_cmap('jet'), colorbar=True, label='population')
	plt.legend()
	plt.show()
	'''

	'''
	# scatter_matrix
	from pandas.tools.plotting import scatter_matrix
	attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
	scatter_matrix(copy_data[attributes], figsize=(12, 8))
	plt.show()
	'''
	
	copy_data["rooms_per_household"] = copy_data["total_rooms"] / copy_data["households"]
	copy_data["bedrooms_per_room"] = copy_data["total_bedrooms"] / copy_data["total_rooms"]
	copy_data["population_per_household"] = copy_data["population"] / copy_data["households"]
	corr_matrix = copy_data.corr()
	print(corr_matrix['median_house_value'].sort_values(ascending=False))
	

df_house = read_data()
#compare_random_split_and_stratified_split(df_house)
visulation_and_discover(df_house)

