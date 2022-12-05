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
X_test= X_test[:,cols]

from openbox import sp

# Define Search Space
space = sp.Space()
lam1 = sp.Real("lam1", 0.001, 3000, default_value=0.2,log=True)
lam2 = sp.Real("lam2", 0.001, 3000, default_value=0.2,log=True)
lam3 = sp.Real("lam3", 0.00001, 3000, default_value=0.2,log=True)
lam5 = sp.Real("lam4", 0.001, 3000, default_value=0.2,log=True)
lam4 = sp.Real("lam5", 0.00001, 3000, default_value=0.2,log=True)
learning_rate = sp.Real("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
space.add_variables([lam1, lam2,lam3,lam4,lam5])
print(learning_rate)

import numpy as np
from sklearn.metrics import mean_squared_error

def GAM_obj(config):

    lam1,lam2,lam3,lam4,lam5 = config['lam1'],config['lam2'],config['lam3'],config['lam4'],config['lam5'] 
    gam = LinearGAM(lam = [lam1,lam2,lam3,lam4,lam5]).fit(X_train,y_train)
    y_pred = gam.predict(X_test)
    mse =  mean_squared_error(y_test,y_pred)
    print(mse)
    return mse 

from openbox import Optimizer


opt = Optimizer(
    GAM_obj,
    space,
    max_runs=50,
    surrogate_type='gp',
    time_limit_per_trial=100,
    task_id='quick_start',
)
history = opt.run()
print(history)
