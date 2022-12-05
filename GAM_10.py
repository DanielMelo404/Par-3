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


from openbox import sp

# Define Search Space
space = sp.Space()
lam1 = sp.Real("lam1", 0.001, 3000, default_value=0.2,log=True)
lam2 = sp.Real("lam2", 0.001, 3000, default_value=0.2,log=True)
lam3 = sp.Real("lam3", 0.00001, 3000, default_value=0.2,log=True)
lam4 = sp.Real("lam4", 0.001, 3000, default_value=0.2,log=True)
lam5 = sp.Real("lam5", 0.00001, 3000, default_value=0.2,log=True)
lam6 = sp.Real("lam6", 0.001, 6000, default_value=0.2,log=True)
lam7 = sp.Real("lam7", 0.001, 6000, default_value=0.2,log=True)
lam8 = sp.Real("lam8", 0.00001, 3000, default_value=0.2,log=True)
lam9 = sp.Real("lam9", 0.001, 3000, default_value=0.2,log=True)
lam10 = sp.Real("lam10", 0.00001, 6000, default_value=0.2,log=True)
lam11 = sp.Real("lam11", 0.001, 3000, default_value=0.2,log=True)
lam12 = sp.Real("lam12", 0.001, 3000, default_value=0.2,log=True)
lam13 = sp.Real("lam13", 0.00001, 3000, default_value=0.2,log=True)
lam14 = sp.Real("lam14", 0.001, 3000, default_value=0.2,log=True)
lam15 = sp.Real("lam15", 0.00001, 3000, default_value=0.2,log=True)
lam16 = sp.Real("lam16", 0.001, 3000, default_value=0.2,log=True)
lam17 = sp.Real("lam17", 0.001, 3000, default_value=0.2,log=True)
lam18 = sp.Real("lam18", 0.00001, 3000, default_value=0.2,log=True)
lam19 = sp.Real("lam19", 0.001, 3000, default_value=0.2,log=True)

learning_rate = sp.Real("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
space.add_variables([lam1,lam2,lam3,lam4,lam5,lam6,lam7,lam8,lam9,lam10, lam11,lam12,lam13,lam14,lam15,lam16, lam17,lam18,lam19])

import numpy as np
from sklearn.metrics import mean_squared_error

def GAM_obj(config):

    lam1,lam2,lam3,lam4,lam5,lam6,lam7,lam8,lam9,lam10, lam11,lam12,lam13,lam14,lam15, lam16,lam17,lam18,lam19 =\
        config['lam1'],config['lam2'],config['lam3'],config['lam4'],config['lam5'], \
        config['lam6'],config['lam7'],config['lam8'],config['lam9'],config['lam10'], \
        config['lam11'],config['lam12'],config['lam13'],config['lam14'],config['lam15'] ,\
        config['lam16'],config['lam17'],config['lam18'],config['lam19']
            
    gam = LinearGAM(lam = [lam1,lam2,lam3,lam4,lam5,lam6,lam7,lam8,lam9,lam10, lam11,lam12,lam13,lam14,lam15, lam16,lam17,lam18,lam19]).fit(X_train,y_train)
    y_pred = gam.predict(X_test)
    mse =  mean_squared_error(y_test,y_pred)
    print(mse)
    return mse 

from openbox import Optimizer


opt = Optimizer(
    GAM_obj,
    space,
    max_runs=100,
    surrogate_type='gp',
    time_limit_per_trial=100,
    task_id='quick_start',
)
history = opt.run()
print(history)
