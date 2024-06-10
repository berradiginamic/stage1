import pandas as pd

# Chemin vers les fichiers CSV
chemin_client = "C:/Users/berra/Desktop/Projets/client.csv"
chemin_client_information = "C:/Users/berra/Desktop/Projets/client_information.csv"
chemin_order = "C:/Users/berra/Desktop/Projets/order_split.csv"
chemin_order_product = "C:/Users/berra/Desktop/Projets/order_product_split.csv"
chemin_folder = "C:/Users/berra/Desktop/Projets/folder.csv"

# Charger les données
client_data = pd.read_csv(chemin_client)
client_information_data = pd.read_csv(chemin_client_information)
order_data = pd.read_csv(chemin_order)
order_product_data = pd.read_csv(chemin_order_product)
folder_data = pd.read_csv(chemin_folder)

# Supprimer les colonnes vides
client_data = client_data.dropna(axis=1, how='all')
client_information_data = client_information_data.dropna(axis=1, how='all')
order_data = order_data.dropna(axis=1, how='all')
order_product_data = order_product_data.dropna(axis=1, how='all')
folder_data = folder_data.dropna(axis=1, how='all')

# Enregistrer les nouvelles versions des tables nettoyées
client_data.to_csv("client_data_cleaned.csv", index=False)
client_information_data.to_csv("client_information_data_cleaned.csv", index=False)
order_data.to_csv("order_data_cleaned.csv", index=False)
order_product_data.to_csv("order_product_data_cleaned.csv", index=False)
folder_data.to_csv("folder_data_cleaned.csv", index=False)