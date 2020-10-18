#! /bin/bash

rm /data/*.csv
rm /data/clean/*.csv

echo "1/3 ==> Scrapping des fichiers depuis www.data.gouv"
python main.py

echo "2/3 ==> Merge des fichiers"
python mergefiles.py
