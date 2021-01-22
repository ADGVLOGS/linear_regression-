# FuelConsumption ADG
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
%matplotlib inline

df = pd.read_csv("FuelConsumption.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


print("Predict CO2EMISSIONS from ENGINESIZE")
x = int(input())
a= regr.intercept_.astype(int)
b= regr.coef_.astype(int)
y=  a + b * x 
print('CO2EMISSIONS is: ',y)


