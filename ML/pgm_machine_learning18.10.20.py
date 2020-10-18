#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# # DATA FILES SOURCE




vehicules= pd.read_csv('/home/fitec/PROJET/out_clean/vehicules.csv',dtype=str)
usagers=pd.read_csv('/home/fitec/PROJET/out_clean/usagers.csv',dtype=str)
charac=pd.read_csv('/home/fitec/PROJET/out_clean/caracteristiques.csv',dtype=str)
lieux=pd.read_csv('/home/fitec/PROJET/out_clean/lieux.csv',dtype=str)





vehicules.info()
#vehicules['catv']=vehicules['catv'].astype('category')
vehicules['catv']=vehicules['catv'].astype('int64')




usagers.info()





usagers[['place','catu','grav','sexe','trajet','secu']]=usagers[['place','catu','grav','sexe','trajet','secu']].astype('int64')

usagers['age']=usagers['age'].astype('float64')





lieux.info()





lieux[['catr','circ','surf','situ']]=lieux[['catr','circ','surf','situ']].astype('int64')


charac.info()

charac[['lum','agg','atm']]=charac[['lum','agg','atm']].astype('int64')


# # MISE EN COMMUN DES DONNEES

df_once=charac.merge(lieux,on=['Num_Acc','year']).drop(['adr'],axis=1)
df_dbl=usagers.merge(vehicules,on=['Num_Acc','num_veh','year'])





df_rdc=df_once.merge(df_dbl,on=['Num_Acc','year'])





df_rdc.shape





print(df_rdc.head())





df_rdc[['an','mois','h24','cp','lat','long','year']]=df_rdc[['an','mois','h24','cp','lat','long','year']].astype('float64')





df_rdc['lat']=df_rdc.loc[:,'lat']/100000





df_rdc['long']=df_rdc.loc[:,'long']/100000


df_rdc.isna().sum()*100/len(df_rdc)





# # EXPLORATION




## analyses bivariées





pd.crosstab(usagers["dc"],charac["lum"],margins=True)





pd.crosstab(usagers["dc"],charac["agg"],margins=True)




pd.crosstab(usagers["dc"],charac["atm"],margins=True)


pd.crosstab(usagers["dc"],lieux["catr"],margins=True)



pd.crosstab(usagers["dc"],lieux["circ"],margins=True)


pd.crosstab(usagers["dc"],lieux["surf"],margins=True)



pd.crosstab(usagers["dc"],lieux["situ"],margins=True)

pd.crosstab(usagers["dc"],vehicules["catv"],margins=True)


pd.crosstab(usagers["dc"],usagers["place"],margins=True)


pd.crosstab(usagers["dc"],usagers["catu"],margins=True)

pd.crosstab(usagers["dc"],usagers["secu"],margins=True)

pd.crosstab(usagers["dc"],usagers["sexe"],margins=True)


pd.crosstab(df_rdc["dc"],df_rdc["mois"],margins=True)



# MATRICES DE CORRELATION


print(df_rdc.corr())

# HISTOGRAMME DES AGES
df_rdc.hist(column='age')


# HISTOGRAMME DE L'AGE SELON LE SEXE
df_rdc.hist(column='age',by='sexe')



df_rdc.boxplot(column='age',by='sexe')


#scatterplot : age vs. grav
df_rdc.plot.scatter(x='grav',y='age')


# HISTOGRAMME du mois de l acc
plt.figure(figsize=(12,6))
df_rdc.mois.hist(rwidth=0.75,alpha =0.50, color= 'blue')
plt.title('fréquence d'accident par mois,fontsize= 30)
plt.grid(False)
plt.xlabel('month jan-dec' , fontsize = 20)
plt.ylabel('Accident count' , fontsize = 15)

# HISTOGRAMME DE L'l heure de l acc
df_rdc.hist(column='mois', by='dc')


plt.figure(figsize=(12,6))
df_rdc.h24.hist(rwidth=0.75,alpha =0.50, color= 'blue')
plt.title('Time of the day/night',fontsize= 30)
plt.grid(False)
plt.xlabel('Time 0-24 hours' , fontsize = 20)
plt.ylabel('Accident count' , fontsize = 15)


df_rdc.h24.mean()


df_rdc['dc'].value_counts()



plt.rc("font", size=14)


sns.countplot(x='dc',data=df_rdc,palette='hls')
plt.show()


sns.countplot(x='gps',data=df_rdc,palette='hls')
plt.show()


df_rdc.groupby('sexe').mean()


pd.crosstab(df_rdc["dc"],df_rdc["trajet"],margins=True)


plt.figure(figsize=(10,10))
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df_rdc.lum,df_rdc.dc).plot(kind='bar')
plt.title('dc Frequency for lum Title')
plt.xlabel('lum')
plt.ylabel('Frequency of dc')
#plt.savefig('purchase_fre_job')
plt.show()

get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df_rdc.sexe,df_rdc.dc).plot(kind='bar')
plt.title('dc Frequency for lum Title')
plt.xlabel('lum')
plt.ylabel('Frequency of dc')


get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df_rdc.gps,df_rdc.dc).plot(kind='bar')
plt.title('dc Frequency for gps Title')
plt.xlabel('gps')
plt.ylabel('Frequency of dc')


plt.figure(figsize=(10,10))
get_ipython().run_line_magic('matplotlib', 'inline')
pd.crosstab(df_rdc.surf,df_rdc.dc).plot(kind='bar')
plt.title('dc Frequency for lum Title')
plt.xlabel('lum')
plt.ylabel('Frequency of dc')

#### plt.scatter(df_rdc.Long,df_rdc.Lat,c = df_rdc.dc)


# #  Test sur la moyenne

# TEST DE STUDENT POUR LES EXPLICATIVES QUANTI

from scipy import stats


stats.ttest_ind(df_rdc.loc[df_rdc.dc=='1','age'],df_rdc.loc[df_rdc.dc=='0','age'])

stats.ttest_ind(df_rdc.loc[df_rdc.dc=='1','cp'],df_rdc.loc[df_rdc.dc=='0','cp'])


stats.ttest_ind(df_rdc.loc[df_rdc.dc=='1','h24'],df_rdc.loc[df_rdc.dc=='0','h24']) #, equal_var=False)


# # ==============    dichotomisation variables catégorielle  ============# 

#  EXPLICATIVES QUALITATIVES
x_quali=df_rdc[['catv','trajet','secu','catr','lum']]


#  one hot encoding
x_dicho = pd.get_dummies(x_quali.astype(str))


data=x_dicho


data[['dc','age','h24','year','agg']]=df_rdc[['dc','age','h24','year','agg']]


data.info()


data.dc.value_counts()


#data.dc.value_counts(normalize=True)

# EXPLICATIVES --feaures
x_var= data.loc[:, data.columns != 'dc']


# In[44]:


# endogène --> labels
y=data.dc


#x_var.info()


# VISUALISE on Y/X
plt.scatter(df_rdc.age,data.dc)

plt.title('distribution du décès' )
plt.xlabel('age')
plt.ylabel('décès')
plt.show()


# # prediction en mode  echantillon apprentissage - test

# TRAINING & TEST DATA 
X_train, X_test, y_train, y_test = train_test_split(x_var, y, test_size=0.3, random_state=1,stratify=data.dc)
df_train, df_test= train_test_split(data, test_size=0.3, random_state=1,stratify=data.dc)

df_train.info()

X_train.shape


X_test.shape


# # =======================  SKlearn ==============================

from sklearn import preprocessing
from sklearn.metrics import classification_report

#instanciation
lr = LogisticRegression(penalty='none',solver='newton-cg')
#X_train=X_train.head(100)

#y_train=y_train.head(100)



#normalisation
stds = preprocessing.StandardScaler()
#transformation
X_Train = stds.fit_transform(X_train)
X_test= stds.fit_transform(X_test)


lr.fit(X_Train,y_train)


X_train.shape

#affichage des coefficients
#print(pd.DataFrame({"var":X_train.columns,"coef":lr.coef_[0]}))


# PREDICTION Y
y_pred = lr.predict(X_test)

# PERFORMANCES


#taux de reconnaissance --> 575257+953/total
print(metrics.accuracy_score(y_test,y_pred))


#Sensibilité 
print(metrics.recall_score(y_test,y_pred,pos_label='0'))

#taux d'erreur --> 14987+1653/total
print(1.0 - metrics.accuracy_score(y_test,y_pred))


#précision – 93/(8+93)
print(metrics.precision_score(y_test,y_pred,pos_label='1'))


y_pred = lr.predict(X_test)
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=y_pred)
print("Accuracy", round(accuracy_score(y_pred, y_test)*100,2))
print(sk_report)
pd.crosstab(y_test, y_pred, rownames=['observé'], colnames=['Prédit'], margins=True)


#  Optimisation par Méthode --> Hyperparameter tuning)
from sklearn.linear_model import LogisticRegressionCV
lr = LogisticRegressionCV(cv=3, random_state=0, max_iter=100)
# Fit the model on the trainng data.
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=y_pred)
print("Accuracy", round(accuracy_score(y_pred, y_test)*100,2))
print(sk_report)
pd.crosstab(y_test, y_pred, rownames=['Observé'], colnames=['Prédit'], margins=True)


#Sensibilité 
print(metrics.recall_score(y_test,y_pred,pos_label='1'))


# #  =================Arbre Decision ============================

from sklearn.tree import DecisionTreeClassifier
arbreFirst=DecisionTreeClassifier(min_samples_split=30,min_samples_leaf=10) #,max_leaf_nodes=20


# Construction de l'arbre

arbreFirst.fit(X_train,y_train)


from sklearn.tree import plot_tree
plt.figure(figsize=(15,15))
plot_tree(arbreFirst,feature_names=list(x_var.columns ),filled=True)
plt.show()

#importance des variables
impVarFirst={"Variable":x_var.columns,"Importance":arbreFirst.feature_importances_}
print(pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False))


#prédiction sur l'échantillon test
pred = arbreFirst.predict(X_test)

#distribution des predictions

print(np.unique(pred,return_counts=True))

# MATRICE DE CONFUSION 
print(metrics.confusion_matrix(y_test,pred))


# # Indicateur de performance

#taux de reconnaissance --> 575257+953/total
print(metrics.accuracy_score(y_test,pred))


# comparaison de l'accuracy entre train & test


# Prediction sur l'echantillon train
pred2 = arbreFirst.predict(X_train)

print(metrics.accuracy_score(y_train,pred2))

# matrice de confusion sur l'echantillon train
print(metrics.confusion_matrix(y_train,pred2))


y_train.shape

#taux d'erreur --> 14987+1653/total
print(1.0 - metrics.accuracy_score(y_test,pred))

#rappel – sensibilité 953/14987+953
print(metrics.recall_score(y_test,pred,pos_label='0'))

#précision – 93/(8+93)
print(metrics.precision_score(y_test,pred,pos_label='1'))



