from bs4 import BeautifulSoup
import requests
import os


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
        with open(fich, 'wb') as file:
            reponse = requests.get(urlFichier)
            file.write(reponse.content)
        print("Téléchargement terminé de : " + nomFich )
        return 1
