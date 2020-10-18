# safeMove
Analyse et prédiction des risques d'accidents corporels liés à la circulation routière

Manuel d'installation

Les étapes suivantes décrives comment déployer l’architecture, scrapper les données, faire des prédictions:


    1. Déploiement de l’infrastructure
	Se positionner sur le dossier bd-infra dans le dossier safeMove: 
             'chmod 777 *.sh' 
             './start.sh' 

	Pour arrêter l'infrastructure: './stop.sh'
       
    2. Création des tables Hive.
        
    • docker-compose exec spark-master bash 
    • Lancer le script spark_Hive.py avec la commande suivante : bin/spark-submit --master yarn /data/script_spark/spark_Hive.py
      
       
    3. Scrapping des données
       Exécuter le script getData.sh : 
    • Se mettre sur le dossier  /data/script_spark
    • Donner les droits au fichier  getData.sh : chmod 777  getData.sh
    • L’executer : ./ getData.sh
       
    4. Nettoyage et import dans Hive
       
       Depuis le spark master exécuter le script spark_clean.py : 
    • cd /usr/spark-2.4.1/
    • bin/spark-submit --master yarn /data/script_spark/spark_clean.py
    • Vérification visuel  de la création de notre base de données ‘pred_accidents’: se connecter à ‘localhost:50070’  et se mettre dans le dossier ‘/user/hive/warehouse’ 
       
    5. Prédiction
       exécuter le script : python ML/pgm_machine_learning.py
