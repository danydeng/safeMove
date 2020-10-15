from bs4 import BeautifulSoup
import requests
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def find_data(url, fileType, dossierData): 
    response = requests.get(url) 
    soup = BeautifulSoup(response.content, "html.parser")

    allDatas = soup.find_all("article", {"class" : "card resource-card"})
    nomFichier = ""    
    lienFichier = ""
    nbrcharge = 0
    nbrIgnore = 0

    for data in allDatas:        
        nomFichier = data.find("div", {"class" : "card-body"}).find("h4", {"class" : "ellipsis"}).text.replace("\n","")
        if fileType in nomFichier:
            lienFichier = data.find("footer").find("div", {"class" : "resource-card-actions btn-toolbar"}).find_all('a', href=True)[0]['href']        
            #print(nomFichier + " ==> " + lienFichier)
            if getCsvFile(nomFichier, lienFichier, dossierData) == 1:
                nbrcharge += 1 #Icrément du nombre de fichier téléchargé
            else :
                nbrIgnore += 1

    print("Nombre de fichiers chargé: " + str(nbrcharge))
    print("Nombre de fichiers ignoré: " + str(nbrIgnore))
    

def getCsvFile(nomFich, urlFichier, dossierSave):
    fich = dossierSave + '/' + nomFich
    ###### rajouter test de vérification de l'existence du fichier et size <> 0 avant chargement
    if os.path.exists(fich) and os.path.getsize(fich) > 0:
        print("Fichier ignoré car existant: " + fich)
        return 0
    else:
        with open(fich.replace('-', '_'), 'wb') as file:
            reponse = requests.get(urlFichier)
            file.write(reponse.content)
        print("Téléchargement terminé de : " + nomFich )
        return 1

""" 
def mergedata(dirPath):
    onlyfiles = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]  #récupère la liste des fichiers du repertoire
    onlyfiles.sort()
    nomFichier = ''
    d = []
    result = []
    for fichier in onlyfiles:  #lecture de tous les fichiers du repertoire
        if "~lock" not in fichier:
            temp = fichier.split('_')
                
            print("chemin: " + dirPath + '/' +fichier)
            
            df = pd.read_csv(os.path.abspath(dirPath + '/' +fichier), encoding='iso-8859-1', dtype=str, error_bad_lines=False)#) #lecture du contenu du csv
            df["year"] = temp[1].split('.')[0] #Ajout colonne année    

            if nomFichier == '':   
                nomFichier = temp[0]  
            elif nomFichier != temp[0]:
                result = pd.concat(d) #merge des dataframes
                print("Ajout fichier : " + dirPath + '/' + nomFichier + ".csv")
                result.to_csv(dirPath + '/' + nomFichier+".csv", index=False) #écriture du fichier de sortie
                d = []
                nomFichier = temp[0]

            d.append(df) #ajout infos dans le dataframe final
            os.remove(os.path.abspath(dirPath + '/' +fichier)) #suppression des fichiers initiaux
    result = pd.concat(d)
    result.to_csv(dirPath + '/' + nomFichier+".csv", index=False) """






