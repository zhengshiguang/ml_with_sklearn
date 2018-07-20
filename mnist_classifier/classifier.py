#coding: utf8
import pandas as pd
import scipy.io

# Import to a python dictionary
mat = scipy.io.loadmat('mnist_all.mat')
print(mat.keys())
print(mat['train0'])
print(mat['__header__'])
print(mat['__version__'])
