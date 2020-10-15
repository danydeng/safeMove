import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import glob

os.chdir("data")

extension = 'csv'
def mergeCarc( fich, extension):  #merge des fichiers caracteristiques
    
    all_carac = [i for i in glob.glob(fich+'*.{}'.format(extension))]
    all_carac.sort()
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f, encoding='iso-8859-1', dtype=str, error_bad_lines=False) for f in all_carac ])
    #export to csv
    combined_csv.to_csv( fich+".csv", index=False, encoding='utf-8-sig')    
    for i in all_carac: 
        os.remove(i) 

mergeCarc('vehicules',extension)
mergeCarc('caracteristiques',extension)
mergeCarc('usagers',extension)
mergeCarc('lieux',extension)

os.chdir("..")