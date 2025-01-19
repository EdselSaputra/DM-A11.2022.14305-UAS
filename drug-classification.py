# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import streamlit as st

# Load the dataset
data = pd.read_csv('dataset/drug_consumption.csv')

# Streamlit app title
st.title("Analisis Pola Konsumsi Narkotika Cannabis berdasarkan Skor Psikologis Individu")

# Pilihan kolom obat
drug_columns = [col for col in data.columns if col not in ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity']]
drug_column = 'Cannabis'

# Pilih skor psikologis
psychological_scores = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']

# Filter dataset untuk kolom yang relevan
selected_columns = psychological_scores + [drug_column]
df_selected = data[selected_columns].copy()


# +
# Konversi kolom konsumsi obat menjadi numerik untuk klasifikasi
def map_drug_usage(value):
    # Map drug usage levels to numerical values
    usage_mapping = {'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6}
    return usage_mapping.get(value, None)

df_selected[drug_column] = df_selected[drug_column].map(map_drug_usage)
# -

# Drop entri dengan nilai null yang dihasilkan dari pemetaan
df_selected = df_selected.dropna()

df_selected[drug_column] = df_selected[drug_column].squeeze()
df_selected['Cannabis_Binary'] = df_selected[drug_column].apply(lambda x: 1 if x > 0 else 0)

# EDA
if st.checkbox("Tampilkan Exploratory Data Analysis (EDA)"):
    # Distribusi target
    st.subheader("Distribusi Penggunaan Obat (Biner)")
    count_fig = plt.figure(figsize=(8, 6))
    sns.countplot(x='Cannabis_Binary', data=df_selected, palette='viridis')
    plt.title('Distribusi Penggunaan Obat (Biner)')
    plt.xlabel('Penggunaan Obat (Biner)')
    plt.ylabel('Jumlah')
    st.pyplot(count_fig)

    # Distribusi skor psikologis
    for score in psychological_scores:
        st.subheader(f"Distribusi {score}")
        score_fig = plt.figure(figsize=(8, 6))
        sns.histplot(df_selected[score], kde=True, color='blue', bins=30)
        plt.title(f'Distribusi {score}')
        plt.xlabel(score)
        plt.ylabel('Frekuensi')
        st.pyplot(score_fig)

X = df_selected[psychological_scores]
y = df_selected['Cannabis_Binary']

# Handle class imbalance menggunakan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# +
# Pilih model
model_choice = st.selectbox("Pilih Model", ["Random Forest", "Logistic Regression"], index=0)

if model_choice == "Random Forest":
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    clf = grid_search.best_estimator_
else:
    # Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
# -

# Prediksi
if st.button("Jalankan Analisis"):
    y_pred = clf.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Hasil Evaluasi Model")
    st.write(f"Akurasi: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))


    # Feature Importances (hanya untuk Random Forest)
    if model_choice == "Random Forest":
        st.subheader("Feature Importances")
        feature_importances = pd.Series(clf.feature_importances_, index=psychological_scores)
        feature_fig = plt.figure(figsize=(8, 6))
        feature_importances.sort_values().plot(kind='barh', color='skyblue')
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        st.pyplot(feature_fig)

# Cross-Validation
if st.checkbox("Tampilkan Cross-Validation"):
    cv_scores = cross_val_score(clf, X_resampled, y_resampled, cv=5)
    st.write("Cross-Validation Scores:", cv_scores)
    st.write("Mean CV Score:", cv_scores.mean())


