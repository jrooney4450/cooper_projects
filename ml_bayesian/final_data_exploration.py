import math
import numpy as np
from scipy.stats import multivariate_normal
import scipy
import matplotlib.pyplot as plt   
from scipy.io import loadmat
import csv
import sklearn
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import random
import pandas as pd

if __name__ == "__main__":

    # Data obtained through kaggle, see here: 
    # https://www.kaggle.com/new-york-city/nyc-property-sales
    data_path = 'nyc-rolling-sales.csv'

    # Import as DataFrame using Pandas
    df = pd.read_csv(data_path)

    # # Here are a bunch of useful pd methods
    # print(df.loc[3:4]) # Return numbered row, can be indexed
    # print(df['SALE PRICE']) # Returns column
    # print(df['SALE PRICE'].values) # Return values of column, can index from here or manipulate
    # print(df.info()) # Get meta-data about columns
    # print(df.columns) # Get list of columns
    # print(df.head()) # Preview of first 5 rows
    # print(df.tail(2)) # Preview of last indices
    # print(df.describe().loc['mean'])

    # Remove sale price as np.array
    sales = df['SALE PRICE'].values

    # Replace strings with '-' values to zeros
    for i in range(len(sales)):
        if sales[i].strip() == '-':
            sales[i] = 0

    # convert ssales data from string to numeric
    sales = pd.to_numeric(sales)

    # find and remove indices where price is too low or 0
    print('size before {}'.format(df.shape))
    drop_index = []
    for i in range(len(sales)):
        if sales[i] < 1000:
            drop_index.append(i)
    df = df.drop(drop_index, axis=0)
    print('size after  {}'.format(df.shape))