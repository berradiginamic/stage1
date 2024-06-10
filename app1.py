import streamlit as st
import pandas as pd

# Chargement des données depuis les fichiers CSV
@st.cache
def load_csv_data(file_path):
    return pd.read_csv(file_path)

# Chemin vers les fichiers CSV
file_paths = {
    "client": "C:/Users/berra/Desktop/Projets/client.csv",
    "client_information": "C:/Users/berra/Desktop/Projets/client_information.csv",
    "order": "C:/Users/berra/Desktop/Projets/order_split.csv",
    "order_product": "C:/Users/berra/Desktop/Projets/order_product_split.csv",
    "folder": "C:/Users/berra/Desktop/Projets/folder.csv"
}

# Affichage des 10 premières lignes et des colonnes disponibles pour chaque table
for table_name, file_path in file_paths.items():
    st.header(f"Table : {table_name}")
    data = load_csv_data(file_path)
    st.write("Les 10 premières lignes :")
    st.write(data.head(10))
    st.write("Colonnes disponibles :")
    st.write(data.columns)


# Charger les données
data = pd.read_csv("C:/Users/berra/Desktop/Projets/client.csv")

# Identifier les colonnes vides
colonnes_vides = data.columns[data.isnull().all()]

# Supprimer les colonnes vides
data = data.drop(colonnes_vides, axis=1)
st.write(data.head(10))

# Afficher les informations sur les colonnes après nettoyage
st.write("Colonnes après nettoyage :")
st.write(data.columns)