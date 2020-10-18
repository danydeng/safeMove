#! /bin/bash

sudo rm /filRouge/spark/master/*.csv
sudo rm data/*.csv

echo "1/3 ==> Scrapping des fichiers depuis www.data.gouv"
python main.py

echo "2/3 ==> Merge des fichiers"
python mergefiles.py

echo "3/3 ==>Déplacement des fichiers dans /filRouge/spark/master/"
sudo mv data/*.csv /filRouge/spark/master/
