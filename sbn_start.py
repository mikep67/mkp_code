snip
# Load data from 'https://coded2.herokuapp.com/datavizpandas/london2018.csv'
# and print it
# First import libraries numpy, and pandas and matplotlib.pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the above url using pandas read_csv
# Refer: pandas import csv from url
# http://pythonfiddle.com/pandas-import-csv-from-url/
import io
import requests

url = 'https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv'
s = requests.get(url).content
ds = pd.read_csv(io.StringIO(s.decode('utf-8')))
#print(ds.describe())
#print(ds)

mtcars = ds

# print(weather.iloc[0:10,0:10])
# print(mtcars.head(10))

# print(mtcars)

mtcars.describe()
