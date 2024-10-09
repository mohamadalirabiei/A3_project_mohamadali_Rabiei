# -*- coding: utf-8 -*-
#start
"""
Created on Tue Oct  8 16:51:20 2024

@author: asus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import mean_absolute_percentage_error #MAPE
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 
'''
in dade ha marbut be qeimate khanehaye california mibashad k bar asase 8 moalefe tayin sodeand
dar marhale aval data ha az ketabkhane sklearn import va check shode and k dadeye tekrari va ya
dadeye khali nadashte bashand
'''
#step 0 --> data cleaning
data=fetch_california_housing()
x=data.data
y=data.target
a=pd.DataFrame(x)
b=pd.DataFrame(y)
a.describe()
b.describe
a.info()
b.info()
data.feature_names
data.target_names
a.duplicated()
a.dropna(inplace=True)
b.dropna(inplace=True)
a.drop_duplicates(inplace=True)

#step 1 --> x o y ra misazim
x1=np.array(a)
y1=np.array(b)

#step 2 --> kfold
kf=KFold(n_splits=10,shuffle=True,random_state=42)

#STEP 3 --> Tayine model ha
model1=LinearRegression()

model2=KNeighborsRegressor()
my_params2= { 'n_neighbors':[1,2,3,4,5,6,11],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }
gs2=GridSearchCV(model2,my_params2,cv=kf,scoring='neg_mean_absolute_percentage_error')

model3=DecisionTreeRegressor()
my_params3={ 'max_depth':[1,2,3,4,5,6,7,11]}
gs3=GridSearchCV(model3, my_params3,cv=kf,scoring='neg_mean_absolute_percentage_error')

model4=RandomForestRegressor()
my_params4={ 'n_stimators':[1,2,3,4,5,6,7,8,9,11] ,
            'max_features':[1,2,3,4,5,6,7,8,9,11] }
gs5=GridSearchCV(model4, my_params4,cv=kf,scoring='neg_mean_absolute_percentage_error')

model5=SVR()
my_params5={'kernel':['poly','rbf','linear'],
           'C':[0.001,0.01,1]}
gs5=GridSearchCV(model5, my_params5,cv=kf,scoring='neg_mean_absolute_percentage_error')

model=[LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor(),SVR()]















