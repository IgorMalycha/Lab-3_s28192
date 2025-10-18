#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py
Kompletny pipeline:
 - eksploracja danych (statystyki + wykresy)
 - czyszczenie i inżynieria cech
 - trenowanie 3 modeli (Linear, Ridge, RandomForest)
 - optymalizacja RandomForest przez GridSearchCV
 - szczegółowe logi w konsoli
 - zapis artefaktów: wykresy, model_results.csv, best_model.pkl
Nie generuje PDF — raport tworzysz ręcznie (Readme.pdf).
"""

import os
import time
import joblib
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -----------------------------
# Settings
# -----------------------------
DATA_PATH = "CollegeDistance.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUT_DIR = "."  # katalog do zapisu artefaktów
PLOTS = {
    "score_dist": os.path.join(OUT_DIR, "score_distribution.png"),
    "corr": os.path.join(OUT_DIR, "correlation_heatmap.png"),
    "box_income": os.path.join(OUT_DIR, "boxplot_score_by_income.png"),
}
RESULTS_CSV = os.path.join(OUT_DIR, "model_results.csv")
BEST_MODEL_FILE = os.path.join(OUT_DIR, "best_model.pkl")

# -----------------------------
# Helpers
# -----------------------------
def safe_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def print_div():
    print("\n" + "=" * 70 + "\n")

def print_section(title: str):
    print_div()
    print(f"### {title}")
    print_div()

def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float,float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return r2, mae, mse, rmse

# -----------------------------
# 1) Wczytanie danych
# -----------------------------
print_section("Wczytywanie danych")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Nie znaleziono pliku danych: {DATA_PATH}")
data = pd.read_csv(DATA_PATH)
print(f"Ścieżka: {DATA_PATH}")
print(f"Liczba rekordów: {len(data)}")
print(f"Liczba kolumn: {len(data.columns)}")
print("Nazwy kolumn:", list(data.columns))
print("\nPodgląd (5 wierszy):")
print(data.head().to_string(index=False))

# -----------------------------
# 2) Eksploracja i czyszczenie
# -----------------------------
print_section("Eksploracja danych (braki, typy, statystyki)")
total_cells = data.shape[0] * data.shape[1]
total_missing = data.isnull().sum().sum()
print(f"Brakujące przed czyszczeniem: {total_missing} ({total_missing/total_cells*100:.3f} % komórek)")
print("\nBraki wg kolumn (przed czyszczeniem):")
print(data.isnull().sum()[data.isnull().sum() > 0].to_string())

# Standardowe "błędne" oznaczenia -> NaN
data = data.replace([" ", "NA", "N/A", "na", "NaN", "None", ""], np.nan)

# Jeżeli są kolumny z wartościami liczonymi jako string, spróbujmy rzutować numerycznie
numeric_guesses = ["unemp", "wage", "distance", "tuition", "education", "score"]
for col in numeric_guesses:
    if col in data.columns:
        # jeśli typ object -> próbujemy konwersji na float
        if data[col].dtype == "object":
            data[col] = pd.to_numeric(data[col].astype(str).str.strip(), errors='coerce')

# Elastyczne mapowanie kolumn boolowych
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
existing_bool_cols = [c for c in bool_cols if c in data.columns]
print(f"\nKolumny logiczne wykryte (próba mapowania): {existing_bool_cols}")

for col in existing_bool_cols:
    # mapuj na stringi lowercase -> mapuj -> coerce -> zostaw NaN jeśli nie pasuje
    data[col] = data[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    # Jeśli kolumna po mapowaniu jest cały NaN (np. była 0/1 jako liczba), spróbuj bez konwersji:
    if data[col].isnull().all():
        # spróbuj rzutowania numeric
        try:
            data[col] = pd.to_numeric(data[col].astype(str), errors='coerce')
        except Exception:
            pass
    # jeżeli nadal NaN istnieją, wypełnij medianą (0 lub 1)
    n_missing = data[col].isnull().sum()
    if n_missing > 0:
        median_val = data[col].median()
        # jeśli median is nan (np. cała kolumna nan), ustaw 0
        if np.isnan(median_val):
            median_val = 0
        data[col].fillna(median_val, inplace=True)
        print(f"Uzupełniono {n_missing} braków w '{col}' wartością mediany: {median_val}")

# Raport braków po mapowaniu booli i rzutowaniu
total_missing_after_bool = data.isnull().sum().sum()
print(f"\nBraki po wstępnej konwersji bool i konwersji numerycznej: {total_missing_after_bool}")

# Imputacja podstawowa: numeryczne medianą, kategoryczne trybem
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = data.select_dtypes(include=['object']).columns.tolist()
print(f"\nRozpoznane zmienne numeryczne ({len(num_cols)}): {num_cols}")
print(f"Rozpoznane zmienne kategoryczne ({len(cat_cols)}): {cat_cols}")

# pokaż statystyki opisowe
print_section("Statystyki opisowe (data.describe())")
pd.set_option("display.max_rows", 200)
print(data.describe().T.to_string())

# Zapis statystyk do zmiennej (możesz wkleić do Readme)
desc = data.describe().T

# Tworzenie wykresów eksploracyjnych
print_section("Generowanie wykresów eksploracyjnych (zapisywane do plików)")
safe_mkdir(OUT_DIR)

# 1) Rozkład targetu score
if 'score' in data.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(data['score'].dropna(), kde=True)
    plt.title("Rozkład zmiennej docelowej: score")
    plt.xlabel("score")
    plt.tight_layout()
    plt.savefig(PLOTS["score_dist"])
    plt.close()
    print(f"Zapisano: {PLOTS['score_dist']}")

# 2) Korelacja (numeryczne)
if len(num_cols) >= 2:
    plt.figure(figsize=(10,8))
    corr = data[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Heatmapa korelacji (zmienne numeryczne)")
    plt.tight_layout()
    plt.savefig(PLOTS["corr"])
    plt.close()
    print(f"Zapisano: {PLOTS['corr']}")

# 3) Boxplot score vs income (jeśli istnieje)
if 'income' in data.columns and 'score' in data.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x='income', y='score', data=data)
    plt.title("Score w zależności od poziomu dochodu (income)")
    plt.tight_layout()
    plt.savefig(PLOTS["box_income"])
    plt.close()
    print(f"Zapisano: {PLOTS['box_income']}")

# -----------------------------
# 3) Inżynieria cech
# -----------------------------
print_section("Inżynieria cech i przygotowanie danych")

# Usuń id jeśli występuje
if 'rownames' in data.columns:
    data.drop(columns=['rownames'], inplace=True)
    print("Usunięto kolumnę 'rownames' (identyfikator).")

# Jeśli istnieją kategorie tekstowe o zbyt wielu poziomach — rozważ agregację (przykład: rare -> 'other')
# (tutaj nie robimy agregacji automatycznej, ale można dodać jeśli potrzeba)

# Przykład: utworzenie cechy dodatkowej (opcjonalnie) - interakcja wage * distance
if 'wage' in data.columns and 'distance' in data.columns:
    data['wage_x_distance'] = data['wage'] * data['distance']
    print("Dodano cechę 'wage_x_distance' (wage * distance).")

# Finalna lista cech
TARGET = 'score'
if TARGET not in data.columns:
    raise KeyError(f"Brakuje kolumny docelowej '{TARGET}' w danych")

X = data.drop(columns=[TARGET])
y = data[TARGET]

# Zaktualizuj listy typów
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Finalne zmienne numeryczne ({len(num_features)}): {num_features}")
print(f"Finalne zmienne kategoryczne ({len(cat_features)}): {cat_features}")

# -----------------------------
# 4) Podział na zbiór treningowy/testowy
# -----------------------------
print_section("Podział danych")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Rozmiar zestawu treningowego: {X_train.shape} (≈{len(X_train)/len(data)*100:.1f}%)")
print(f"Rozmiar zestawu testowego:     {X_test.shape} (≈{len(X_test)/len(data)*100:.1f}%)")

# -----------------------------
# 5) Pipeline: imputacja -> scaling -> encoding
# -----------------------------
print_section("Budowa pipeline'u (imputacja, skalowanie, encoding)")

# Używamy SimpleImputer w transformerach żeby mieć pewność braku NaN we wnętrzu pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
], remainder='drop')

print("Preprocessor zbudowany. Num features:", len(num_features), " Cat features:", len(cat_features))

# -----------------------------
# 6) Modele do treningu (minimum 3)
# -----------------------------
print_section("Wybór modeli i trening")
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE) if hasattr(Ridge, 'random_state') else Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
}

# Safety check: czy w X_train/test nie ma NaN (przed pipeline imputacją)
nan_train = X_train.isnull().sum().sum()
nan_test = X_test.isnull().sum().sum()
print(f"Węzeł kontrolny - braków w surowych X_train: {nan_train}, X_test: {nan_test}")
if nan_train > 0 or nan_test > 0:
    print("Uwaga: surowe X zawiera NaN - pipeline ma imputery które je uzupełnią, jednak warto to mieć na uwadze.")

results = []
model_objects = {}

for name, estimator in models.items():
    t0 = time.time()
    print(f"\nTrenuję model: {name}")
    pipe = Pipeline(steps=[('preproc', preprocessor), ('model', estimator)])
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    # predykcja i metryki (test)
    y_pred = pipe.predict(X_test)
    r2, mae, mse, rmse = metrics_report(y_test, y_pred)
    # cross-val score on training for stability
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    results.append({
        "Model": name,
        "R2_test": float(r2),
        "MAE_test": float(mae),
        "MSE_test": float(mse),
        "RMSE_test": float(rmse),
        "CV_R2_mean": float(np.mean(cv_scores)),
        "CV_R2_std": float(np.std(cv_scores)),
        "Train_time_s": float(train_time)
    })
    model_objects[name] = pipe

    print(f"Wyniki dla {name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"Cross-val R2 (5-fold): mean={np.mean(cv_scores):.4f} std={np.std(cv_scores):.4f}")
    print(f"Czas treningu: {train_time:.2f}s")

# -----------------------------
# 7) Zapis podstawowych wyników
# -----------------------------
print_section("Zapis wyników podstawowych")
results_df = pd.DataFrame(results).sort_values(by="R2_test", ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))
results_df.to_csv(RESULTS_CSV, index=False)
print(f"Wyniki zapisano do: {RESULTS_CSV}")

# -----------------------------
# 8) Optymalizacja najlepszego modelu (GridSearchCV dla RandomForest)
# -----------------------------
print_section("Optymalizacja najlepszego modelu (GridSearchCV dla RandomForest)")

# Wybierz RandomForest jako kandydat do tuningu (jeśli jest w modelach)
if "Random Forest" not in model_objects:
    print("Random Forest nie był trenowany wcześniej — pomijam tunning.")
else:
    # Zbuduj pipeline z preproc i RF (again)
    rf_pipeline = Pipeline(steps=[('preproc', preprocessor),
                                  ('model', RandomForestRegressor(random_state=RANDOM_STATE))])

    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
    }

    print("Param grid:", param_grid)
    grid = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    t0 = time.time()
    grid.fit(X_train, y_train)
    grid_time = time.time() - t0

    print(f"GridSearch zakończony w {grid_time:.1f}s")
    print("Najlepsze parametry:", grid.best_params_)
    best_rf = grid.best_estimator_

    # Ewaluacja najlepszej wersji RF
    y_pred_best = best_rf.predict(X_test)
    r2_b, mae_b, mse_b, rmse_b = metrics_report(y_test, y_pred_best)
    cv_scores_best = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    tuned_result = {
        "Model": "RandomForest (GridSearchCV tuned)",
        "R2_test": float(r2_b),
        "MAE_test": float(mae_b),
        "MSE_test": float(mse_b),
        "RMSE_test": float(rmse_b),
        "CV_R2_mean": float(np.mean(cv_scores_best)),
        "CV_R2_std": float(np.std(cv_scores_best)),
        "Train_time_s": float(grid_time)
    }

    print(f"Tuned RF: R2_test={r2_b:.4f}, MAE={mae_b:.4f}, RMSE={rmse_b:.4f}")
    # dodaj wynik do tabeli i zapisz flattened
    results_df = pd.concat([results_df, pd.DataFrame([tuned_result])], ignore_index=True)
    results_df = results_df.sort_values(by="R2_test", ascending=False).reset_index(drop=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print("Zaktualizowane wyniki zapisano do:", RESULTS_CSV)

    # Zapis najlepszego modelu (tuned)
    joblib.dump(best_rf, BEST_MODEL_FILE)
    print(f"Zapisano najlepszy (strojony) model: {BEST_MODEL_FILE}")

# -----------------------------
# 9) Finalne podsumowanie
# -----------------------------
print_section("Finalne podsumowanie i rekomendacje")
print("Podsumowanie modeli (posortowane wg R2_test):")
print(results_df.to_string(index=False))

best_row = results_df.iloc[0]
print_div()
print(f"Najlepszy model: {best_row['Model']}")
print(f"R2_test = {best_row['R2_test']:.4f}, MAE = {best_row['MAE_test']:.4f}, RMSE = {best_row['RMSE_test']:.4f}")
print_div()

print("\nArtefakty wygenerowane:")
for k, v in PLOTS.items():
    if os.path.exists(v):
        print(f" - {v}")
print(f" - {RESULTS_CSV}")
if os.path.exists(BEST_MODEL_FILE):
    print(f" - {BEST_MODEL_FILE}")
else:
    print(" - (Brak pliku best_model.pkl — możliwe, że GridSearch nie był uruchamiany)")

print("\nInstrukcja do Readme.pdf (co wkleić):")
print("- Krótkie wprowadzenie / cel projektu")
print("- Opis danych (kolumny) + statystyki (wklej output data.describe())")
print("- Wykresy: score_distribution.png, correlation_heatmap.png, boxplot_score_by_income.png")
print("- Opis przygotowania danych: imputacja, mapowanie bool, utworzone cechy (wage_x_distance jeśli występuje)")
print("- Modele: wymień 3 modele, podaj uzasadnienie wyboru")
print("- Metryki: wklej tabelę model_results.csv (R2, MAE, MSE, RMSE, CV mean/std)")
print("- Wynik optymalizacji: najlepsze parametry GridSearch i porównanie przed/po")
print("- Wnioski i dalsze kroki")

print_section("Koniec")
