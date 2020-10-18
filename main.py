import getOpenData
import os
extension = ".csv"
url = "https://www.data.gouv.fr/fr/datasets/base-de-donnees-accidents-corporels-de-la-circulation/"
dossier = "data"


getOpenData.find_data(url, extension, dossier)
""" os.popen('python mergefiles.py')
print("Fin merge files\n")
print("copie fichiers vers spark master\n")
os.popen('sudo mv data/*.csv /filRouge/spark/master/')
print("Fin de la copie\n")
#getOpenData.mergedata(dossier)
#print(os.path.abspath("data/caracteristiques_2011.csv")) """