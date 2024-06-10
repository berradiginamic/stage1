import pandas as pd
from sqlalchemy import create_engine

# Connexion à la base de données PostgreSQL
database_url = "postgresql://salam_report:NuY11PIKgZ3A@197.140.18.127:6432/dbsalamprod"
engine = create_engine(database_url)

# Liste des tables à exporter
tables = ["client", "client_information", "folder", "order_product_split", "order_split"]  # Ajoutez ici toutes les tables nécessaires

for table in tables:
    # Lire les données de la table
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, engine)

    # Exporter les données en fichier CSV
    csv_file = f"C:/Users/berra/Desktop/Projets/{table}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')

    print(f"Table {table} exportée en {csv_file}")