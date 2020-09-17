from bs4 import BeautifulSoup
import requests



def find_data(url, fileType, dossierData): 
    response = requests.get(url) 
    soup = BeautifulSoup(response.content, "html.parser")

    allDatas = soup.find_all("article", {"class" : "card resource-card"})
    nomFichier = ""    
    lienFichier = ""
    count = 0
    for data in allDatas:        
        nomFichier = data.find("div", {"class" : "card-body"}).find("h4", {"class" : "ellipsis"}).text.replace("\n","")
        if fileType in nomFichier:
            lienFichier = data.find("footer").find("div", {"class" : "resource-card-actions btn-toolbar"}).find_all('a', href=True)[0]['href']        
            #print(nomFichier + " ==> " + lienFichier)
            getCsvFile(nomFichier, lienFichier, dossierData)
            count = count + 1
    print("Nombre de fichiers : " + str(count))
    

def getCsvFile(nomFich, urlFichier, dossierSave):

    ###### rajouter test de vérification de l'existence du fichier et size <> 0 avant chargement

    with open(dossierSave + '/' + nomFich, 'wb') as file:
        reponse = requests.get(urlFichier)
        file.write(reponse.content)
    print("Téléchargement terminé de : " + nomFich )




#print(metiers)
