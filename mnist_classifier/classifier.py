#coding: utf8
import pandas as pd
import scipy.io

# Import to a python dictionary
mat = scipy.io.loadmat('mnist_all.mat')
print(mat['train0'])
