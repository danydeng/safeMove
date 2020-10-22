from pyspark import SparkContext, SparkConf
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, HiveContext
import os

sparkSession =SparkSession.builder.master("local[*]").appName("appName")\
    .config("hive.metastore.uris", "thrift://hive-metastore:9083")\
    .config("spark.sql.warehouse.dir", "hdfs://namenode:8020/user/hive/warehouse")\
    .enableHiveSupport().getOrCreate()

sqlContext = HiveContext(sparkSession)
requete = ""
db_name = "pred_accidents"
############### création de la BD ################################
sqlContext.sql("CREATE DATABASE IF NOT EXISTS " + db_name)

sqlContext.sql("use " + db_name)

############### création des tables ################################
requete = 'CREATE TABLE IF NOT EXISTS caracteristiques_tab ( `Num_acc` string , `an` INT, `mois` INT, `jour` INT, `heure` string, `lum` string, `agg` string, `atm` string, `com` string, `adr` string, `gps` string, `lat` string, `long` string, `dep` string, `year` INT, `cp` string, `h24` string ) ROW FORMAT DELIMITED FIELDS TERMINATED BY ","'
sqlContext.sql(requete)

requete = 'CREATE TABLE IF NOT EXISTS usagers_tab (Num_acc string, place string, catu string , grav string, sexe string, trajet string, secu string, an_nais int, num_veh string, year INT, dc string, age int) ROW FORMAT DELIMITED FIELDS TERMINATED BY ","'
sqlContext.sql(requete)

requete = 'CREATE TABLE IF NOT EXISTS vehicules_tab  ( Num_Acc string , catv string , num_veh string, year int) ROW FORMAT DELIMITED FIELDS TERMINATED BY ","'
sqlContext.sql(requete)

requete = 'CREATE TABLE IF NOT EXISTS lieux_tab ( Num_Acc string , catr string , circ string , surf string, situ string, year int) ROW FORMAT DELIMITED FIELDS TERMINATED BY ","'
sqlContext.sql(requete)






df_load = sqlContext.sql('SHOW TABLES')
df_load.show()
#print(df_load.show())