# ======================================
#  ZADANIE 1 - MODEL PREDYKCYJNY
#  Autor: Student XYZ
#  Plik: model.py
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

print("========================")
print("  ZADANIE 1 - MODEL PREDYKCYJNY")
print("========================\n")

# ===============================
# 1️⃣ Wczytanie danych
# ===============================
print("📂 Wczytywanie danych...")

data = pd.read_csv("CollegeDistance.csv")

print(f"Liczba rekordów: {len(data)}")
print(f"Liczba kolumn: {len(data.columns)}\n")

print("Podgląd danych:")
print(data.head(), "\n")

# ===============================
# 2️⃣ Wstępna analiza danych i czyszczenie
# ===============================
print("🔍 Analiza brakujących danych i czyszczenie...")

# Liczba braków przed czyszczeniem
missing_before_total = data.isnull().sum().sum()
print(f"Brakujące wartości przed czyszczeniem: {missing_before_total}")

# Zamiana błędnych wartości tekstowych na NaN
data = data.replace([" ", "NA", "N/A", "na", "NaN", "None"], np.nan)

# Kolumny logiczne
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})
    data[col] = data[col].fillna(data[col].median())

# Wypełnianie braków numerycznych medianą
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        median_val = data[col].median()
        data[col].fillna(median_val, inplace=True)
        print(f"🧮 Wypełniono braki w kolumnie '{col}' medianą: {median_val:.3f}")

# Wypełnianie braków kategorycznych trybem
for col in data.select_dtypes(include=["object"]).columns:
    if data[col].isnull().sum() > 0:
        mode_val = data[col].mode()[0]
        data[col].fillna(mode_val, inplace=True)
        print(f"🗂️ Wypełniono braki w kolumnie '{col}' trybem: {mode_val}")

missing_after_total = data.isnull().sum().sum()
print(f"\nBrakujące wartości po czyszczeniu: {missing_after_total}")

if missing_after_total == 0:
    print("✅ Wszystkie dane kompletne!\n")
else:
    print("⚠️ Nadal występują NaN-y po czyszczeniu!\n")

# ===============================
# 3️⃣ Przygotowanie danych
# ===============================
print("🧹 Przygotowanie danych...")

# Usunięcie kolumny identyfikatora
if "rownames" in data.columns:
    data.drop(columns=["rownames"], inplace=True)
    print("🗑️ Kolumna 'rownames' została usunięta.\n")

# Zmienna docelowa
target = "score"
X = data.drop(columns=[target])
y = data[target]

# Identyfikacja typów zmiennych
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

print("🔢 Zmienne numeryczne:", num_features)
print("🔠 Zmienne kategoryczne:", cat_features, "\n")

# ===============================
# 4️⃣ Podział danych
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("📈 Podział danych:")
print(f" - Zbiór treningowy: {len(X_train)} próbek ({len(X_train)/len(data)*100:.1f}%)")
print(f" - Zbiór testowy: {len(X_test)} próbek ({len(X_test)/len(data)*100:.1f}%)\n")

# ===============================
# 5️⃣ Transformacje i pipeline
# ===============================
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

# ===============================
# 6️⃣ Trening i ewaluacja modeli
# ===============================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Walidacja danych (czy nie zawierają NaN)
assert not X_train.isnull().any().any(), "❌ Zbiór treningowy zawiera NaN!"
assert not X_test.isnull().any().any(), "❌ Zbiór testowy zawiera NaN!"

results = []

print("🤖 Trenowanie modeli...\n")

for name, model in models.items():
    print(f"➡️  Trening modelu: {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)

    # Predykcja i metryki
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results.append({
        "Model": name,
        "R²": r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })

    print(f"📊 Wyniki {name}:")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAE:  {mae:.4f}")
    print(f"   MSE:  {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}\n")

# ===============================
# 7️⃣ Podsumowanie wyników
# ===============================
print("📋 PODSUMOWANIE MODELI\n")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

best_model = results_df.loc[results_df["R²"].idxmax()]
print("\n🏆 Najlepszy model:", best_model["Model"])
print(f"   R² = {best_model['R²']:.4f}, MAE = {best_model['MAE']:.4f}")

# Zapis wyników do pliku (opcjonalnie)
results_df.to_csv("model_results.csv", index=False)
print("\n📁 Wyniki zapisano do pliku: model_results.csv")

print("\n✅ Proces zakończony sukcesem!\n")
