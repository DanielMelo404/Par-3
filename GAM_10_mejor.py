
lam1                    =609.058983   
lam10                   =3000.000000  
lam11                   =12.446086    
lam12                   =6.999498     
lam13                   =0.733207     
lam14                   =689.982889   
lam15                   =197.089242   
lam16                   =0.016831     
lam17                   =0.501379     
lam18                   =59.207062    
lam19                   =0.095920     
lam2                    =310.524535   
lam3                    =3000.000000  
lam4                    =2526.968409  
lam5                    =0.016507     
lam6                    =3000.000000  
lam7                    =3000.000000  
lam8                    =0.086510     
lam9                    =35.009964    

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

X_test =  loadtxt('X_shuff_test.csv', delimiter=',')
X_train = loadtxt('X_shuff_train.csv', delimiter=',')
y_test =  loadtxt('y_shuff_test.csv', delimiter=',')
y_train = loadtxt('y_shuff_train.csv', delimiter=',')
y_all =  loadtxt('y_shuffled.csv', delimiter=',')
X_all = loadtxt('X_shuffled.csv', delimiter=',')


cols = np.array([ 1,  4, 14, 16, 18])


from openbox import sp
import numpy as np
from sklearn.metrics import mean_squared_error

 
gam = LinearGAM(lam = [lam1,lam2,lam3,lam4,lam5,lam6,lam7,lam8,lam9,lam10, lam11,lam12,lam13,lam14,lam15, lam16,lam17,lam18,lam19]).fit(X_train,y_train)
y_pred = gam.predict(X_test)
print(np.count_nonzero(y_pred<0))
mse =  mean_squared_error(y_test,y_pred)

print(mse)


def cval(data, labels, k):
    # data = np.array(data)
    # labels = np.array(labels)
    data_split = np.vsplit(data, k)
    label_split = np.vsplit(labels, k)

    # training classifier
    score = 0
    for i in range(k):
        data_train = np.concatenate((data_split[:i]+data_split[i+1:]), axis = 1)
        labels_train = np.concatenate((label_split[:i]+label_split[i+1:]), axis = 1)
        data_test = data_split[i]
        labels_test = label_split[i]
        score += eval(data_train, data_test, labels_train, labels_test)
    return score/k  

def eval(d_train,d_test,l_train,l_test):
    gam = LinearGAM(lam = [lam1,lam2,lam3,lam4,lam5,lam6,lam7,lam8,lam9,lam10, lam11,lam12,lam13,lam14,lam15, lam16,lam17,lam18,lam19]).fit(d_train,l_train)
    y_pred = gam.predict(d_test)
    mse =  mean_squared_error(l_test,y_pred)
    return mse


print(cval(X_all,y_all))
