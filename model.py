# ======================================
#  ZADANIE 1 - MODEL PREDYKCYJNY
#  Autor: Student XYZ
#  Plik: model_predykcyjny.py
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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
# 1️⃣ Wczytanie i eksploracja danych
# ===============================
print("📂 Wczytywanie danych...")

data = pd.read_csv("CollegeDistance.csv")

print(f"Liczba rekordów: {len(data)}")
print(f"Liczba kolumn: {len(data.columns)}\n")

print("Podgląd danych:")
print(data.head(), "\n")

# Statystyki opisowe
print("📊 Statystyki opisowe:")
print(data.describe(include='all'), "\n")

# Wizualizacja rozkładu zmiennej docelowej
plt.figure(figsize=(6,4))
sns.histplot(data['score'], kde=True, color='skyblue')
plt.title("Rozkład zmiennej docelowej 'score'")
plt.show()

# Macierz korelacji
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji zmiennych numerycznych")
plt.show()

# ===============================
# 2️⃣ Czyszczenie danych
# ===============================
print("🔍 Czyszczenie danych...")

# Zamiana błędnych wartości na NaN
data = data.replace([" ", "NA", "N/A", "na", "NaN", "None"], np.nan)

# Kolumny binarne
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    data[col] = (
        data[col]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Wypełnianie braków numerycznych medianą
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Wypełnianie braków kategorycznych trybem
for col in data.select_dtypes(include=["object"]).columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

print(f"Brakujące wartości po czyszczeniu: {data.isnull().sum().sum()}")

# ===============================
# 3️⃣ Inżynieria cech
# ===============================
print("\n🧠 Inżynieria cech...")

# Usunięcie zbędnych kolumn
if "rownames" in data.columns:
    data.drop(columns=["rownames"], inplace=True)

# Tworzenie nowych cech
if "distance" in data.columns and "income" in data.columns:
    data["income_per_distance"] = data["income"] / (data["distance"] + 1)
    print("➕ Dodano cechę: income_per_distance")

# Logarytmowanie zmiennej 'distance' (jeśli istnieje)
if "distance" in data.columns:
    data["log_distance"] = np.log1p(data["distance"])

# ===============================
# 4️⃣ Podział na zbiory treningowy i testowy
# ===============================
target = "score"
X = data.drop(columns=[target])
y = data[target]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Zbiór treningowy: {len(X_train)} próbek")
print(f"Zbiór testowy: {len(X_test)} próbek\n")

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
# 6️⃣ Trening modeli i uzasadnienie
# ===============================
print("🤖 Trenowanie modeli...\n")

# Uzasadnienie (opis do raportu)
"""
MODELE:
- Linear Regression → prosty model bazowy, służy jako punkt odniesienia.
- Ridge Regression → regresja liniowa z regularyzacją L2, zapobiega przeuczeniu.
- Random Forest → model nieliniowy, dobrze radzi sobie z interakcjami i brakami liniowości.
"""

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
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

results_df = pd.DataFrame(results)
print(results_df)

# Wykres porównawczy
plt.figure(figsize=(8,5))
sns.barplot(x="Model", y="R²", data=results_df, palette="pastel")
plt.title("Porównanie modeli wg R²")
plt.show()

# ===============================
# 7️⃣ Optymalizacja najlepszego modelu
# ===============================
best_model_name = results_df.loc[results_df["R²"].idxmax(), "Model"]
print(f"\n🏆 Najlepszy model: {best_model_name}\n")

if best_model_name == "Random Forest":
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30]
    }

    grid = GridSearchCV(
        Pipeline(steps=[('preprocessor', preprocessor),
                        ('model', RandomForestRegressor(random_state=42))]),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    print("🔍 Najlepsze parametry:", grid.best_params_)
    print(f"Średni R² (CV): {grid.best_score_:.4f}")

# ===============================
# 8️⃣ Walidacja krzyżowa i podsumowanie
# ===============================
pipeline_best = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', RandomForestRegressor(random_state=42))])

scores = cross_val_score(pipeline_best, X, y, cv=5, scoring='r2')
print(f"\n📈 Średni wynik R² w walidacji krzyżowej: {scores.mean():.4f}")

# Zapis wyników
results_df.to_csv("model_results.csv", index=False)
print("\n📁 Wyniki zapisano do pliku: model_results.csv")

print("\n✅ Proces zakończony sukcesem!\n")
