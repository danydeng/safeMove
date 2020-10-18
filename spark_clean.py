#!/usr/bin/python
import subprocess
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, HiveContext
#from hdfs import InsecureClient

sparkSession =SparkSession.builder.master("local[*]").appName("appName")\
    .config("hive.metastore.uris", "thrift://hive-metastore:9083")\
    .config("spark.sql.warehouse.dir", "hdfs://namenode:8020/user/hive/warehouse")\
    .enableHiveSupport().getOrCreate()

# importer les fichiers brut dans des dataFrame
vehicules= pd.read_csv('/data/vehicules.csv',dtype=str)
usagers=pd.read_csv('/data/usagers.csv',dtype=str)
crtrtic=pd.read_csv('/data/caracteristiques.csv',dtype=str)
lieux=pd.read_csv('/data/lieux.csv',dtype=str)

################################## DATA VEHICULES ################################################################
# definition type des variables
vehicules['catv']=vehicules['catv'].astype('category')

# ajout de year suite modif merge scrap
vehicules["year"]=vehicules['Num_Acc'].str[0:4]

vehicules.shape
vehicules.info()

#vehicules.describe()


vehicules=vehicules.loc[:,['Num_Acc','catv','num_veh','year']]
pd.isnull(vehicules["catv"]).sum()

print(vehicules["catv"].isna().sum())

#################################    DATA USAGERS   #################################################################
usagers.shape
usagers=usagers.drop(['locp','actp','etatp'],axis=1)

usagers[['place','catu','grav','sexe','trajet','secu']]=usagers[['place','catu','grav','sexe','trajet','secu']].astype('category')
usagers['an_nais']=usagers['an_nais'].astype('float64')


# ajout de year suite modif merge scrap
usagers["year"]=usagers['Num_Acc'].str[0:4]

#verification de la modalité '0' pour place
usagers['place'].value_counts()

usagers.loc[:,'grav'].value_counts().sum()

usagers.isna().sum()*100/len(usagers)

usagers.loc[:,'an_nais'].describe()
an_miss=usagers[usagers.loc[:,'an_nais'].isna()]

# Remplacer les valeurs manquantes de années nais par la moyenne
usagers.loc[:,'an_nais'].mean()
usagers['an_nais'].fillna(usagers['an_nais'].mean(), inplace=True)

usagers.isna().sum()*100/len(usagers)

# 2% de la variable secu est mqt ---> on peut envisager une imputation par le mode
usagers.loc[:,'secu'].value_counts()

# la modalité '0' est définie comme valeur manquante pour l'ensemble des variables 
# on peut les imputer par le mode
usagers.loc[usagers.place=='0','place']=usagers.loc[:,'place'].mode()

# Imputation secu, trajet & place
usagers['secu'].fillna(usagers['secu'].mode()[0], inplace=True)
usagers['place'].fillna(usagers['place'].mode()[0], inplace=True)
usagers['trajet'].fillna(usagers['trajet'].mode()[0], inplace=True)

usagers.isna().sum()*100/len(usagers)

# Personnes DCD
usagers['dc']=0

usagers.loc[usagers['grav']=='2','dc']=1

usagers['age']=pd.to_datetime('today').year - usagers['an_nais']

usagers.head()

################################### DATA LIEUX   ##################################################################
# ajout de year suite modif merge scrap
lieux["year"]=lieux['Num_Acc'].str[0:4]

lieux.head()
lieux.info()
lieux.shape

# vérification de la modalités '0' pour surf
lieux['surf'].value_counts()

# la modalité '0' est définie comme valeur manquante pour l'ensemble des variables 
# on peut les imputer par le mode
lieux.loc[lieux.surf=='0','surf']=lieux.loc[:,'surf'].mode()

lieux[['catr','circ','surf','situ']]=lieux[['catr','circ','surf','situ']].astype('category')

lieux=lieux[['Num_Acc','catr','circ','surf','situ','year']]
lieux.head()

# % de valeurs manquantes
lieux.isna().sum()*100/len(lieux)

lieux['surf'].fillna(lieux['surf'].mode()[0], inplace=True)
lieux['situ'].fillna(lieux['situ'].mode()[0], inplace=True)
lieux['circ'].fillna(lieux['circ'].mode()[0], inplace=True)
lieux['catr'].fillna(lieux['catr'].mode()[0], inplace=True)

#######################################  DATA Caractéristiques   ##################################################

# ajout de year suite modif merge scrap
crtrtic["year"]=crtrtic['Num_Acc'].str[0:4]

crtrtic.shape
crtrtic.info()
crtrtic[['lum','agg','atm']]=crtrtic[['lum','agg','atm']].astype('category')


crtrtic.head()

crtrtic=crtrtic.drop(['int','col'],axis=1)
# Imputation atm
crtrtic['atm'].fillna(crtrtic['atm'].mode()[0], inplace=True)

# exploration latitude/longitude mqt 
lat_miss=crtrtic[crtrtic.loc[:,'lat'].isna()]
crtrtic.isna().sum()*100/len(crtrtic)

# remplacer les - dans long par nan
crtrtic.loc[crtrtic.long=='-','long']=''

#  taux élevé de DM pour lat/long/gps ~ 47 % 
lat_miss.shape
lat_miss.head(10)

# preparation de la colonne commune [0:1]
print(crtrtic['com'].head())
print(crtrtic['dep'][0:3].head())

# Pour les commune a 2 chiffres pour <100 : rajouter un '0' ou  '00 avant pour le cp ---> 75012 au lieu de 7512 ou 75002
crtrtic['com'] = crtrtic['com'].apply(lambda x: '0' + x if len(str(x)) ==2  else ('00' + x if len(str(x)) == 1 else x))

crtrtic['cp']=crtrtic['dep'].str[0:2] + crtrtic['com']

# recoupage de l'heure sur 24
crtrtic['hrmn'] = crtrtic['hrmn'].apply(lambda x: '0' + x if len(str(x)) == 3  else ('00' + x if len(str(x)) == 2 else ( '000' + x if len(str(x)) == 1 else x)))
crtrtic['h24']=crtrtic['hrmn'].str[0:2]
crtrtic.head()
crtrtic.isna().sum()*100/len(crtrtic)


usagers.to_csv('/data/clean/usagers_clean.csv',index=False)
vehicules.to_csv('/data/clean/vehicules_clean.csv',index=False)
crtrtic.to_csv('/data/clean/caracteristiques_clean.csv',index=False)
lieux.to_csv('/data/clean/lieux_clean.csv',index=False)

sqlContext = HiveContext(sparkSession)
db_name = "pred_accidents"
sqlContext.sql("use " + db_name)

sqlContext.sql('LOAD DATA LOCAL INPATH "/data/clean/usagers_clean.csv" OVERWRITE INTO TABLE usagers_tab')
sqlContext.sql('LOAD DATA LOCAL INPATH "/data/clean/vehicules_clean.csv" OVERWRITE INTO TABLE vehicules_tab')
sqlContext.sql('LOAD DATA LOCAL INPATH "/data/clean/caracteristiques_clean.csv" OVERWRITE INTO TABLE caracteristiques_tab')
sqlContext.sql('LOAD DATA LOCAL INPATH "/data/clean/lieux_clean.csv" OVERWRITE INTO TABLE lieux_tab')

df_load = sqlContext.sql( "select * from caracteristiques_tab limit 10")
df_load.show()
