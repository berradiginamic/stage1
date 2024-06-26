import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from prophet import Prophet

# Chargement des données depuis les tables modifiées
client_data = pd.read_csv("client_data_cleaned.csv")
client_information_data = pd.read_csv("client_information_data_cleaned.csv")
order_data = pd.read_csv("order_data_cleaned.csv")
order_product_data = pd.read_csv("order_product_data_cleaned.csv")
folder_data = pd.read_csv("folder_data_cleaned.csv")

# Ajoutez la sélection de l'analyse dans la barre latérale
analysis = st.sidebar.selectbox("Choisissez une analyse",
                                ["Segmentation des clients", "Prévision des annulations de commandes",
                                 "Prévision des chiffres d'affaires", "Scoring des clients pour acceptation"])


# Ajoutez une condition pour l'analyse de la segmentation des clients
if analysis == "Segmentation des clients":
    st.header("Segmentation des clients")

    # Utilisez les données des clients et des informations sur les clients pour la segmentation
    data = pd.merge(client_data, client_information_data, on='id')
    # Mettez à jour la liste des colonnes à utiliser en fonction de celles disponibles dans vos données
    features = data[['monthly_salaryretrait', 'net_salaryrena', 'other_income', 'mortgage_amount']]

    st.write("Forme des caractéristiques avant imputation:", features.shape)

    # Vérifiez les colonnes avec des valeurs manquantes
    missing_columns = features.columns[features.isnull().any()].tolist()
    st.write("Colonnes avec des valeurs manquantes:", missing_columns)

    # Imputer les valeurs manquantes
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)

    # Convertir en DataFrame avec les noms de colonnes originaux
    features_imputed_df = pd.DataFrame(features_imputed, columns=features.columns[:features_imputed.shape[1]])

    st.write("Forme des caractéristiques après imputation:", features_imputed_df.shape)

    # Si des colonnes sont manquantes, les réajouter avec des valeurs imputées appropriées
    for col in missing_columns:
        if col not in features_imputed_df.columns:
            features_imputed_df[col] = features[col].mean()

    # Réorganiser les colonnes dans l'ordre original
    features_imputed_df = features_imputed_df[features.columns]

    # Affichez les informations après imputation
    st.write("Statistiques descriptives après imputation:")
    st.write(features_imputed_df.describe())

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_imputed_df)

    # Utilisez la méthode du coude pour déterminer le nombre optimal de clusters
    inertia = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.title('Méthode du coude')
    st.pyplot(plt)

    # Utilisez K-means pour la segmentation
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(features_scaled)

    # Affichez une paire de tracés pour visualiser les clusters
    sns.pairplot(data, hue='Cluster', vars=features.columns)
    st.pyplot()

    # Affichez les résultats de la segmentation
    st.write('Segmentation des clients:')
    st.write(data[['firstname', 'lastname', 'Cluster']])

elif analysis == "Prévision des annulations de commandes":
    st.header("Prévision des annulations de commandes")

    st.write("Colonnes disponibles dans order_data:", order_data.columns)

    # Utilisation des colonnes disponibles pour l'analyse
    if all(col in order_data.columns for col in
           ['id', 'created_at', 'order_type', 'previous_status', 'state', 'payment_type']):
        features = order_data[['id', 'created_at', 'order_type', 'previous_status', 'state', 'payment_type']].dropna()

        # Correspondance correcte des échantillons pour les features et les labels
        order_data['created_at'] = pd.to_datetime(order_data['created_at'])
        order_data['created_at_timestamp'] = order_data['created_at'].astype('int64') // 10 ** 9

        features = order_data[['id', 'created_at_timestamp', 'order_type', 'previous_status', 'state', 'payment_type']]
        features = pd.get_dummies(features, columns=['order_type', 'previous_status', 'state', 'payment_type'])
        features = features.dropna()

        labels = (order_data.loc[features.index, 'state'] == 'cancelled').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write('Rapport de classification:')
        st.text(classification_report(y_test, y_pred))
    else:
        st.write("Certaines colonnes nécessaires ne sont pas disponibles dans order_data.")

elif analysis == "Prévision des chiffres d'affaires":
    st.header("Prévision des chiffres d'affaires")

    if 'created_at' in order_product_data.columns:
        order_product_data['created_at'] = pd.to_datetime(order_product_data['created_at'])
        revenue_data = order_product_data.groupby(order_product_data['created_at'].dt.date)[
            'selling_price'].sum().reset_index()
        revenue_data.columns = ['ds', 'y']

        if not revenue_data.empty:
            model = Prophet()
            model.fit(revenue_data)

            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            fig = model.plot(forecast)
            st.pyplot(fig)

            st.write('Prévisions des chiffres d\'affaires:')
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        else:
            st.write("Les données de prévision des chiffres d'affaires sont vides.")
    else:
        st.write("La colonne 'created_at' n'existe pas dans la table 'order_product_data'.")

elif analysis == "Scoring des clients pour acceptation":
    st.header("Scoring des clients pour acceptation")

    # Utilisation des colonnes disponibles pour l'analyse
    if all(col in folder_data.columns for col in
           ['id', 'created_at', 'folder_state', 'folder_previous_state', 'request_number']):
        features = folder_data[['created_at', 'folder_state', 'folder_previous_state']].dropna()

        # Conversion des dates en timestamps
        features['created_at'] = pd.to_datetime(features['created_at']).astype('int64') // 10 ** 9

        # Convertir les features catégorielles en numériques
        label_encoders = {}
        for column in ['folder_state', 'folder_previous_state']:
            label_encoders[column] = LabelEncoder()
            features[column] = label_encoders[column].fit_transform(features[column])

        labels = (folder_data['folder_state'] == 'accepted').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.write('Rapport de classification:')
        st.text(classification_report(y_test, y_pred))

        # Validation croisée
        scores = cross_val_score(model, features, labels, cv=5)
        st.write("Scores de validation croisée:")
        st.write(scores)
        st.write("Score moyen de validation croisée:", scores.mean())

        # Vérifier l'équilibre des classes
        st.write("Distribution des classes dans les données:", folder_data['folder_state'].value_counts())

        # Inspecter les caractéristiques les plus importantes
        importances = model.feature_importances_
        feature_names = features.columns
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importances = feature_importances.sort_values(by='importance', ascending=False)
        st.write("Importance des caractéristiques:", feature_importances)

        # Analyse des résidus
        errors = X_test[y_test != y_pred]
        st.write("Erreurs de prédiction (s'il y en a):", errors)

        # Examiner les échantillons de validation croisée
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(features):
            X_train_cv, X_test_cv = features.iloc[train_index], features.iloc[test_index]
            y_train_cv, y_test_cv = labels.iloc[train_index], labels.iloc[test_index]
            model_cv = RandomForestClassifier(random_state=42)
            model_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = model_cv.predict(X_test_cv)
            st.write("Rapport de classification pour ce pli:")
            st.text(classification_report(y_test_cv, y_pred_cv))
    else:
        st.write("Certaines colonnes nécessaires ne sont pas disponibles dans folder_data.")
