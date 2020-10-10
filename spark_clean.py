#!/usr/bin/python
import subprocess
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, HiveContext
from hdfs import InsecureClient

sparkSession =SparkSession.builder.master("local[*]").appName("appName")\
    .config("hive.metastore.uris", "thrift://hive-metastore:9083")\
    .config("spark.sql.warehouse.dir", "hdfs://namenode:8020/user/hive/warehouse")\
    .enableHiveSupport().getOrCreate()

vehicules= pd.read_csv('/data/vehicules.csv',dtype=str)
usagers=pd.read_csv('/data/usagers.csv',dtype=str)
crtrtic=pd.read_csv('/data/caracteristiques.csv',dtype=str)
lieux=pd.read_csv('/data/lieux.csv',dtype=str)

vehicules['catv']=vehicules['catv'].astype('category')

vehicules.shape
vehicules.info()
vehicules.tail(10)

vehicules=vehicules.loc[:,['Num_Acc','catv','num_veh','year']]
pd.isnull(vehicules["catv"]).sum()

vehicules.head()

print(vehicules["catv"].isna().sum())

usagers.shape
usagers=usagers.drop(['locp','actp','etatp'],axis=1)

usagers[['place','catu','grav','sexe','trajet','secu']]=usagers[['place','catu','grav','sexe','trajet','secu']].astype('category')
usagers['an_nais']=usagers['an_nais'].astype('float64')

usagers.loc[:,'grav'].value_counts().sum()
usagers.isna().sum()*100/len(usagers)
usagers.loc[:,'an_nais'].describe()

an_miss=usagers[usagers.loc[:,'an_nais'].isna()]
an_miss.head(10)

# Remplacer les valeurs manquantes de age par la moyenne
usagers.loc[:,'an_nais'].mean()
usagers['an_nais'].fillna(usagers['an_nais'].mean(), inplace=True)

#usagers.isna().sum()*100/len(usagers)

pd.crosstab(usagers["grav"],usagers["sexe"],margins=True)
usagers.loc[:,'secu'].value_counts()

# calcul du mode 
usagers.loc[:,'secu'].mode()

# 5% de la variable place est mqt --->  imputation par le mode
usagers.loc[:,'place'].value_counts()

usagers.loc[:,'place'].mode()

# Imputation secu & place
usagers['secu'].fillna(usagers['secu'].mode()[0], inplace=True)
usagers['place'].fillna(usagers['place'].mode()[0], inplace=True)
usagers.isna().sum()*100/len(usagers)

# Personnes DCD
usagers['dc']=0

usagers.loc[usagers['grav']=='2','dc']=1
pd.crosstab(usagers["dc"],usagers["secu"],margins=True)
usagers['age']=pd.to_datetime('today').year - usagers['an_nais']
usagers.head()

################################### DATA LIEUX   ##################################################################

lieux.head()
lieux.info()
lieux.shape

lieux[['catr','circ','surf','situ']]=lieux[['catr','circ','surf','situ']].astype('category')

lieux=lieux[['Num_Acc','catr','circ','surf','situ','year']]
lieux.head()

# % de valeurs manquantes
lieux.isna().sum()*100/len(lieux)

#######################################  DATA Caractéristiques   ##################################################
crtrtic.shape
crtrtic.info()

crtrtic[['lum','agg','atm']]=crtrtic[['lum','agg','atm']].astype('category')
crtrtic.head()

crtrtic=crtrtic.drop(['int','col'],axis=1)
crtrtic.head()
crtrtic.shape
crtrtic.isna().sum()*100/len(crtrtic)

# identification des lignes avec 4,88 % de missing sur tte les variables
num_miss=crtrtic[crtrtic.loc[:,'Num_Acc'].isna()]

num_miss.tail(10)  # 45443 valeur manquante sur ttes les colonnes, pas d'imputation possible --> supprimer ?
# suppression de 45443 obs avec Valeurs mqt partout
crtrtic=crtrtic.loc[crtrtic.loc[:,'Num_Acc'].notna(),:]
crtrtic.shape

# exploration latitude/longitude mqt 
lat_miss=crtrtic[crtrtic.loc[:,'lat'].isna()]

# au vu du taux élevé de DM pour lat/long/gps 47 % il vaut mieux ne pas inclure ces variables
lat_miss.shape
lat_miss.head(10)

# construction de la colonne commune [0:1]
print(crtrtic['com'].head())
print(crtrtic['dep'][0:3].head())

# Pour les commune a 2 chiffres pour <100 : rajouter un '0' ou  '00 avant pour le cp ---> 75012 au lieu de 7512 ou 75002
crtrtic['com'] = crtrtic['com'].apply(lambda x: '0' + x if len(str(x)) ==2  else ('00' + x if len(str(x)) == 1 else x))
crtrtic['cp']=crtrtic['dep'].str[0:2] + crtrtic['com']
crtrtic.head()
crtrtic.tail()

# recoupage de l'heure sur 24
crtrtic['h24']=crtrtic['hrmn'].str[0:2]
crtrtic.head()

usagers.to_csv('/data/usagers_clean.csv',index=False)
vehicules.to_csv('/data/vehicules_clean.csv',index=False)
crtrtic.to_csv('/data/caracteristiques_clean.csv',index=False)
lieux.to_csv('/data/lieux_clean.csv',index=False)

sqlContext = HiveContext(sparkSession)
db_name = "accidents"
sqlContext.sql("use " + db_name)

sqlContext.sql('LOAD DATA LOCAL INPATH "/data/usagers_clean.csv" OVERWRITE INTO TABLE usagers_tab')
sqlContext.sql('LOAD DATA LOCAL INPATH "/data/vehicules_clean.csv" OVERWRITE INTO TABLE vehicules_tab')
sqlContext.sql('LOAD DATA LOCAL INPATH "/data/caracteristiques_clean.csv" OVERWRITE INTO TABLE caracteristiques_tab')
sqlContext.sql('LOAD DATA LOCAL INPATH "/data/lieux_clean.csv" OVERWRITE INTO TABLE lieux_tab')

""" client_hdfs = InsecureClient('http://172.27.1.5:50070')

with client_hdfs.write('/user/hive/vehicules_clean.csv', encoding = 'utf-8') as writer:
 vehicules.to_csv(writer)

with client_hdfs.write('/user/hive/lieux_clean.csv', encoding = 'utf-8') as writer:
 lieux.to_csv(writer)

with client_hdfs.write('/user/hive/usagers_clean.csv', encoding = 'utf-8') as writer:
 usagers.to_csv(writer)

with client_hdfs.write('/user/hive/caracteristiques_clean.csv', encoding = 'utf-8') as writer:
 crtrtic.to_csv(writer) """


df_load = sqlContext.sql( "select * from caracteristiques_tab limit 10")
df_load.show()