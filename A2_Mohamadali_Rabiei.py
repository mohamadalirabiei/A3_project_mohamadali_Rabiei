# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:51:20 2024

@author: asus

fght n_splits ro 5 konid
kf=KFold(n_splits=5,shuffle=True,random_state=42)
va yek rune dg begirid (va socre haye jadid ro bzarid) va dar enteha yek ghesmate report bezarid va begid data ha chi bode va x ha chi bdoan y chi bodd hadaf chi bode
5 ta mdoelo begido moghayese konid kodom bhtrin bode va tamom shdo baram email konid

chra? ---> vbaghty migid n_splits=10 yani data ro 1/10 mikone yani 10% ro var midare b onvane test ama vghty migid n_splits=5 yani 20% hamon 20-25% k goftim baayd vardahste bshe
va hamchenin tedade 5 bar inkaro mikone (crossvalidation) pas zamane kamtari ttool mikeshe har run
moafagh bashid


"""

#-----------Import Libs----------------------

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
from sklearn.preprocessing import MinMaxScaler
'''
in dade ha marbut be qeimate khanehaye california mibashad k bar asase 8 moalefe tayin sodeand
dar marhale aval data ha az ketabkhane sklearn import va check shode and k dadeye tekrari va ya
dadeye khali nadashte bashand
'''

#-----------Import DATA----------------------

data=fetch_california_housing()
x=data.data
y=data.target

#-----------STEP0 : DATA CLEANING----------------------

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
'''

dar khatte 43 va 44 etelaate dade ha gerefte shodand va besurate zir namayesh dade shodand
ke moshakhas shod hameye dade ha dorost va non-null budand va hich objecti darune anha vojud nadasht
ama baraye etminan az inke dade yi khali va ya tekrari nabashad anhara drop kardim.

a.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 8 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       20640 non-null  float64
 1   1       20640 non-null  float64
 2   2       20640 non-null  float64
 3   3       20640 non-null  float64
 4   4       20640 non-null  float64
 5   5       20640 non-null  float64
 6   6       20640 non-null  float64
 7   7       20640 non-null  float64
dtypes: float64(8)

b.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 1 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   0       20640 non-null  float64
dtypes: float64(1)

'''
#-----------Step1 : X and Y ----------------------
x1=np.array(a)
y1=np.array(b)

#-----------Step2 : KFOLD ----------------------
kf=KFold(n_splits=10,shuffle=True,random_state=42)

#-----------Step3 : Model selection ----------------------
#-----------LR ----------------------
model1=LinearRegression()
my_params1= {}
gs1=GridSearchCV(model1,my_params1,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs1.fit(x1,y1)
#-----------KNN ----------------------
model2=KNeighborsRegressor()
my_params2= { 'n_neighbors':[1,2,3,4,5,6,100],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }
gs2=GridSearchCV(model2,my_params2,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs2.fit(x1,y1)
#-----------DT ----------------------
model3=DecisionTreeRegressor()
my_params3={ 'max_depth':[1,2,3,4,5,6,7,100]}
gs3=GridSearchCV(model3, my_params3,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs3.fit(x1,y1)
#-----------RF----------------------
model4=RandomForestRegressor()
my_params4={ 'n_estimators':[1,2,3,4,5,6,7,8,9,100] ,
            'max_features':[1,2,3,4,5,6,7,8,9] }
gs4=GridSearchCV(model4, my_params4,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs4.fit(x1,y1)
#----------SVR ----------------------
model5=SVR()
scaler= MinMaxScaler()
X_scaled=scaler.fit_transform(x1)
my_params5={'kernel':['poly','rbf','linear'],
           'C':[0.0001,0.001,0.01,1,10]}
gs5=GridSearchCV(model5, my_params5,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs5.fit(X_scaled,y1)
#model=[LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor(),SVR()]

#-----------Step4 : best score & param ----------------------
gs1.best_score_    #---->    np.float64(-0.3178265911552671)
gs2.best_score_    #---->    np.float64(-0.4806836705270506)
gs3.best_score_    #---->    np.float64(-0.2503102576177472)
gs4.best_score_    #---->    np.float64(-0.17883512446232863)
gs5.best_score_    #---->    np.float64(-0.22329321703244903)

gs1.best_params_    #---->   no params {}
gs2.best_params_    #---->   {'metric': 'manhattan', 'n_neighbors': 5}
gs3.best_params_    #---->   {'max_depth': 100}
gs4.best_params_    #---->   {'max_features': 4, 'n_estimators': 100}
gs5.best_params_    #---->   {'C': 10, 'kernel': 'rbf'}

print('dar nahayar natije mishavad k modele RF ba 0.17883512446232863 khata va ba max_features: 4, n_estimators: 100 behtarin modele in qesmat ast')


#کد از صفر اصلاح شد و ایرادات و تغیراتی که از سمت شما گزارش شده بود اعمال شد







