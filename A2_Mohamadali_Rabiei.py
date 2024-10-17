# -*- coding: utf-8 -*-
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
from sklearn.preprocessing import MinMaxScaler
'''
in dade ha marbut be qeimate khanehaye california mibashad k bar asase 8 moalefe tayin shode and
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
kf=KFold(n_splits=5,shuffle=True,random_state=42)
'''
hengami k n_splits ro barabare 5 migozarim yani dade hara be 5 qesmat taqsim kardim 
va harbar yek qesmat ra be onvane test barmidarim (20% be onvane test)
'''
#-----------Step3 : Model selection ----------------------
#-----------LR ----------------------
model1=LinearRegression()
my_params1= {}
gs1=GridSearchCV(model1,my_params1,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs1.fit(x1,y1)
#-----------KNN ----------------------
model2=KNeighborsRegressor()
my_params2= { 'n_neighbors':[1,2,3,4,5,6,7,8,10,11,13,15,17,20,50,100],
            'metric':['minkowski'  , 'euclidean' , 'manhattan'] }
gs2=GridSearchCV(model2,my_params2,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs2.fit(x1,y1)
#-----------DT ----------------------
model3=DecisionTreeRegressor()
my_params3={ 'max_depth':[1,2,3,4,5,6,7,10,10,11,13,15,17,20,50,100]}
gs3=GridSearchCV(model3, my_params3,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs3.fit(x1,y1)
#-----------RF----------------------
model4=RandomForestRegressor()
my_params4={ 'n_estimators':[1,2,3,4,5,6,7,8,9,10,11,13,15,17,20,40,100] ,
            'max_features':[1,2,3,4,5,6,7,8,9,10,11,13,15,17,20] }
gs4=GridSearchCV(model4, my_params4,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs4.fit(x1,y1)
#----------SVR ----------------------
model5=SVR()
scaler= MinMaxScaler()
X_scaled=scaler.fit_transform(x1)
my_params5={'kernel':['poly','rbf','linear'],
           'C':[0.0001,0.001,0.01,1,5,2,3,4,6,7,8,9]}
gs5=GridSearchCV(model5, my_params5,cv=kf,scoring='neg_mean_absolute_percentage_error')
gs5.fit(X_scaled,y1)
#model=[LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor(),SVR()]

#-----------Step4 : best score & param ----------------------
gsbest1=gs1.best_score_    #---->    
gsbest2=gs2.best_score_    #---->    
gsbest3=gs3.best_score_    #---->    
gsbest4=gs4.best_score_    #---->    
gsbest5=gs5.best_score_    #---->    

gsbest_params1=gs1.best_params_    #---->   no params {}
gsbest_params2=gs2.best_params_    #---->   
gsbest_params3=gs3.best_params_    #---->   
gsbest_params4=gs4.best_params_    #---->   
gsbest_params5=gs5.best_params_    #---->   

gs_best = [gsbest1, gsbest2, gsbest3, gsbest4, gsbest5] 
best_nahayi = max(gs_best)  

if best_nahayi == gs_best[0]:
   print('dar nahayat natije mishavad k modele LR ba' ,gsbest1, 'khata va ba myparams:',gsbest_params1, 'behtarin modele in qesmat ast')
elif best_nahayi == gs_best[1]:
    print('dar nahayat natije mishavad k modele KNN ba' ,gsbest2, 'khata va ba myparams:',gsbest_params2, 'behtarin modele in qesmat ast')
elif best_nahayi == gs_best[2]:
    print('dar nahayat natije mishavad k modele DT ba' ,gsbest3, 'khata va ba myparams:',gsbest_params3, 'behtarin modele in qesmat ast')
elif best_nahayi == gs_best[3]:
   print('dar nahayat natije mishavad k modele RF ba' ,gsbest4, 'khata va ba myparams:',gsbest_params4, 'behtarin modele in qesmat ast')
elif best_nahayi == gs_best[4]:
   print('dar nahayat natije mishavad k modele SVR ba' ,gsbest5, 'khata va ba myparams:',gsbest_params5, 'behtarin modele in qesmat ast')
else:
    raise TypeError('dade ha qalat and ya modeli vojud nadaraad ke BETAVANAD be dorosti in regression ra anjam dahad')

#report
'''
in dade ha be do daste x va y taqsim bandi shod ke dar anha x barabare moalefe haye khaneha
mesle metrazh, moqeyiate makani, sene khane va ... budand
va y barabare qeimate har khane ast
va in regression be manzure in anjam shod ke betavan ba moalefehaye mozkoor qeimate khane hara pishbini nemud

dar akhar natije shod k modele4 (random forest behtarin pishbini ra beine model haye diar darad)
'''








