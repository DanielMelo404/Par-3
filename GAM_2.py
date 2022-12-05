import sklearn
import openbox
from openbox import run_test
import pygam
from pygam import LinearGAM
from numpy import loadtxt

import numpy as np

import pandas as pd
df = pd.read_excel('data.xlsx')
X = df.iloc[:, 0:19].values
y = df.iloc[:, 19].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_test = loadtxt('X_test.csv', delimiter=',')
X_train = loadtxt('X_train.csv', delimiter=',')
y_test = loadtxt('y_test.csv', delimiter=',')
y_train = loadtxt('y_train.csv', delimiter=',')
cols = np.array([ 1,  4, 14, 16, 18])
X_train= X_train[:,cols]

from openbox import sp

import numpy as np
from sklearn.metrics import mean_squared_error

lam1 = 100
lam2 = 100
lam3 = 100
lam4 = 100
lam5 = 100

gam = LinearGAM(lam = [lam1,lam2,lam3,lam4,lam5]).fit(X_train,y_train)
lams = [lam1,lam2,lam3,lam4,lam5]

y_pred = LinearGAM.predict(X_train)
print(mean_squared_error(y_test,y_pred))
gam.gridsearch(X_train, y_train, lam=lams)


