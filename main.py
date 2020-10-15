import getOpenData
import os
extension = ".csv"
url = "https://www.data.gouv.fr/fr/datasets/base-de-donnees-accidents-corporels-de-la-circulation/"
dossier = "data"


getOpenData.find_data(url, extension, dossier)
#getOpenData.mergedata(dossier)
#print(os.path.abspath("data/caracteristiques_2011.csv"))