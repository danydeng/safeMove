#! /bin/bash

docker-compose up -d

docker-compose exec spark-master bash -c "apt update"

docker-compose exec spark-master bash -c "pip install numpy"

docker-compose exec spark-master bash -c "pip install pandas"

docker-compose exec spark-master bash -c "pip install matplotlib"

docker-compose exec spark-master bash -c "pip install sklearn"

docker-compose exec spark-master bash -c "pip install seaborn"

docker-compose exec spark-master bash -c "pip install glob"

docker-compose exec spark-master bash -c "pip install hdfs"

docker-compose exec spark-master bash -c "cp conf/spark-defaults.conf.template conf/spark-defaults.conf"

docker-compose exec spark-master bash -c 'echo "spark.driver.extraClassPath /data/jar_files/*" >> conf/spark-defaults.conf'

sudo cp -r conf/jar_files /filRouge/spark/master

sudo mkdir /filRouge/spark/master/script_spark

sudo mkdir /filRouge/spark/master/clean

sudo cp ../spark_Hive.py /filRouge/spark/master/script_spark

sudo cp ../spark_clean.py /filRouge/spark/master/script_spark