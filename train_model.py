import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv('german_credit_data.csv')

print("\n================ DATASET ================")

print(df.head())

print("\nJumlah Data :", df.shape[0])
print("Jumlah Kolom :", df.shape[1])

print("\nNama Kolom :")
print(df.columns.tolist())

# ============================================================
# HAPUS KOLOM TIDAK PENTING
# ============================================================

if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# ============================================================
# CEK TARGET
# ============================================================

target_col = 'Risk'

if target_col not in df.columns:

    raise Exception(
        "\nERROR: Kolom 'Risk' tidak ditemukan!\n"
        "Gunakan dataset German Credit yang memiliki kolom Risk."
    )

# ============================================================
# CEK MISSING VALUE
# ============================================================

print("\n================ MISSING VALUE ================")

print(df.isnull().sum())

# ============================================================
# HANDLE MISSING VALUE
# ============================================================

for col in df.select_dtypes(include='object').columns:

    df[col] = df[col].fillna('unknown')

# ============================================================
# ENCODING
# ============================================================

label_encoders = {}

cat_cols = df.select_dtypes(include='object').columns

print("\n================ ENCODING ================")

for col in cat_cols:

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])

    label_encoders[col] = le

    mapping = dict(
        zip(
            le.classes_,
            le.transform(le.classes_)
        )
    )

    print(f"\n{col}")

    print(mapping)

# ============================================================
# SPLIT FITUR & TARGET
# ============================================================

X = df.drop(target_col, axis=1)

y = df[target_col]

print("\n================ FITUR ================")

print(X.columns.tolist())

print("\nDistribusi Target :")

print(y.value_counts())

# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n================ SPLIT DATA ================")

print("Jumlah Train :", X_train.shape[0])

print("Jumlah Test  :", X_test.shape[0])

# ============================================================
# ELBOW METHOD
# ============================================================

print("\n================ ELBOW METHOD ================")

error_rates = []

k_values = range(1, 21)

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    error = 1 - accuracy_score(y_test, pred)

    error_rates.append(error)

    print(f"K = {k} | Error = {error:.4f}")

# ============================================================
# CARI K TERBAIK
# ============================================================

best_k = error_rates.index(min(error_rates)) + 1

print(f"\nK Optimal : {best_k}")

# ============================================================
# TRAIN MODEL FINAL
# ============================================================

knn = KNeighborsClassifier(
    n_neighbors=best_k
)

knn.fit(X_train, y_train)

# ============================================================
# PREDIKSI
# ============================================================

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================ HASIL MODEL ================")

print(f"Akurasi : {accuracy:.2%}")

print("\nClassification Report :")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ============================================================
# BUAT FOLDER STATIC
# ============================================================

os.makedirs('static', exist_ok=True)

# ============================================================
# GRAFIK ELBOW METHOD
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    k_values,
    error_rates,
    marker='o',
    linewidth=2,
    color='blue'
)

plt.axvline(
    x=best_k,
    color='red',
    linestyle='--',
    label=f'K Optimal = {best_k}'
)

plt.title('Elbow Method KNN')

plt.xlabel('Nilai K')

plt.ylabel('Error Rate')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    'static/elbow_method.png'
)

print("\nMenampilkan Grafik Elbow Method...")

plt.show()

plt.close()

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['BAD', 'GOOD']
)

fig, ax = plt.subplots(figsize=(6, 6))

disp.plot(ax=ax)

plt.title('Confusion Matrix')

plt.tight_layout()

plt.savefig(
    'static/confusion_matrix.png'
)

print("\nMenampilkan Confusion Matrix...")

plt.show()

plt.close()

# ============================================================
# ACCURACY CHART
# ============================================================

plt.figure(figsize=(6, 5))

bars = plt.bar(
    ['Accuracy'],
    [accuracy],
    color='green'
)

plt.ylim(0, 1)

plt.title('Model Accuracy')

for bar in bars:

    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 0.02,
        f'{accuracy:.2%}',
        ha='center'
    )

plt.tight_layout()

plt.savefig(
    'static/accuracy_chart.png'
)

print("\nMenampilkan Accuracy Chart...")

plt.show()

plt.close()

# ============================================================
# SIMPAN MODEL
# ============================================================

os.makedirs('model', exist_ok=True)

pickle.dump(
    knn,
    open('model/knn_model.pkl', 'wb')
)

pickle.dump(
    scaler,
    open('model/scaler.pkl', 'wb')
)

pickle.dump(
    label_encoders,
    open('model/label_encoders.pkl', 'wb')
)

pickle.dump(
    best_k,
    open('model/best_k.pkl', 'wb')
)

pickle.dump(
    list(X.columns),
    open('model/feature_columns.pkl', 'wb')
)

# ============================================================
# INFORMASI AKHIR
# ============================================================

print("\n================ FILE TERSIMPAN ================")

print("model/knn_model.pkl")
print("model/scaler.pkl")
print("model/label_encoders.pkl")
print("model/best_k.pkl")
print("model/feature_columns.pkl")

print("\n================ GRAFIK BERHASIL DIBUAT ================")

print("static/elbow_method.png")
print("static/confusion_matrix.png")
print("static/accuracy_chart.png")

print("\n================ TRAINING SELESAI ================")

print("MODEL BERHASIL DITRAIN!")

print("\nJalankan Flask dengan:")

print("py -3.11 app.py")