# ======================================
#  ZADANIE 1 - MODEL PREDYKCYJNY
#  Autor: Student XYZ
#  Plik: model.py
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Konfiguracja wizualizacji
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("  ZADANIE 1 - MODEL PREDYKCYJNY Z PEŁNĄ ANALIZĄ")
print("=" * 60 + "\n")

# ===============================
# 1️⃣ EKSPLORACJA I WSTĘPNA ANALIZA DANYCH (5 pkt)
# ===============================
print("📂 ETAP 1: EKSPLORACJA I WSTĘPNA ANALIZA DANYCH")
print("=" * 60)

# Wczytanie danych
data = pd.read_csv("CollegeDistance.csv")

print(f"Liczba rekordów: {len(data)}")
print(f"Liczba kolumn: {len(data.columns)}")
print(f"\nKolumny w datasecie: {list(data.columns)}\n")

print("Podgląd pierwszych 5 wierszy:")
print(data.head())

# Informacje o typach danych
print("\n📊 Informacje o typach danych:")
print(data.info())

# Statystyki opisowe
print("\n📈 Statystyki opisowe zmiennych numerycznych:")
print(data.describe())

# Analiza brakujących wartości
print("\n🔍 Analiza brakujących wartości:")
missing_data = data.isnull().sum()
missing_percent = (missing_data / len(data)) * 100
missing_df = pd.DataFrame({
    'Kolumna': missing_data.index,
    'Braki': missing_data.values,
    'Procent': missing_percent.values
}).query('Braki > 0')

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("✅ Brak brakujących wartości w surowych danych!")

# WIZUALIZACJA 1: Rozkład zmiennej docelowej
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data['score'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Score')
axes[0].set_ylabel('Częstość')
axes[0].set_title('Rozkład zmiennej docelowej (score)', fontsize=12, fontweight='bold')
axes[0].axvline(data['score'].mean(), color='red', linestyle='--', label=f'Średnia: {data["score"].mean():.2f}')
axes[0].axvline(data['score'].median(), color='green', linestyle='--', label=f'Mediana: {data["score"].median():.2f}')
axes[0].legend()

axes[1].boxplot(data['score'], vert=True)
axes[1].set_ylabel('Score')
axes[1].set_title('Boxplot zmiennej score', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_rozklad_zmiennej_docelowej.png', dpi=300, bbox_inches='tight')
print("\n✅ Wykres zapisany: 01_rozklad_zmiennej_docelowej.png")
plt.close()

# ===============================
# 2️⃣ CZYSZCZENIE DANYCH
# ===============================
print("\n🧹 Czyszczenie danych...")

# Zamiana błędnych wartości tekstowych na NaN
data = data.replace([" ", "NA", "N/A", "na", "NaN", "None"], np.nan)

# Kolumny logiczne
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.lower()
            .map({"yes": 1, "no": 0})
        )
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)
            print(f"✅ Kolumna '{col}' uzupełniona medianą: {data[col].median()}")

# Wypełnianie braków numerycznych medianą
for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        median_val = data[col].median()
        data[col].fillna(median_val, inplace=True)
        print(f"🧮 Wypełniono braki w '{col}' medianą: {median_val:.3f}")

# Wypełnianie braków kategorycznych trybem
for col in data.select_dtypes(include=["object"]).columns:
    if data[col].isnull().sum() > 0:
        mode_val = data[col].mode()[0]
        data[col].fillna(mode_val, inplace=True)
        print(f"🗂️ Wypełniono braki w '{col}' trybem: {mode_val}")

print(f"\nBrakujące wartości po czyszczeniu: {data.isnull().sum().sum()}")

# WIZUALIZACJA 2: Macierz korelacji
numeric_cols = data.select_dtypes(include=[np.number]).columns
correlation_matrix = data[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Macierz korelacji zmiennych numerycznych', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('02_macierz_korelacji.png', dpi=300, bbox_inches='tight')
print("✅ Wykres zapisany: 02_macierz_korelacji.png")
plt.close()

# WIZUALIZACJA 3: Korelacja z zmienną docelową
score_corr = correlation_matrix['score'].sort_values(ascending=False)[1:]

plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in score_corr.values]
plt.barh(score_corr.index, score_corr.values, color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Korelacja ze zmienną score')
plt.title('Korelacja cech ze zmienną docelową', fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('03_korelacja_ze_score.png', dpi=300, bbox_inches='tight')
print("✅ Wykres zapisany: 03_korelacja_ze_score.png")
plt.close()

# WIZUALIZACJA 4: Rozkłady zmiennych numerycznych
num_features = [col for col in numeric_cols if col != 'score']
n_features = len(num_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, col in enumerate(num_features):
    axes[idx].hist(data[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Częstość')
    axes[idx].set_title(f'Rozkład: {col}', fontweight='bold')
    axes[idx].axvline(data[col].mean(), color='red', linestyle='--', alpha=0.7)

for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('04_rozklady_zmiennych.png', dpi=300, bbox_inches='tight')
print("✅ Wykres zapisany: 04_rozklady_zmiennych.png")
plt.close()

# ===============================
# 2️⃣ INŻYNIERIA CECH I PRZYGOTOWANIE DANYCH (5 pkt)
# ===============================
print("\n" + "=" * 60)
print("📐 ETAP 2: INŻYNIERIA CECH I PRZYGOTOWANIE DANYCH")
print("=" * 60)

# Usunięcie kolumny identyfikatora
if "rownames" in data.columns:
    data.drop(columns=["rownames"], inplace=True)
    print("🗑️ Usunięto kolumnę 'rownames'\n")

# Zmienna docelowa
target = "score"
X = data.drop(columns=[target])
y = data[target]

# Identyfikacja typów zmiennych
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

print("🔢 Zmienne numeryczne:", num_features)
print("🔠 Zmienne kategoryczne:", cat_features)

# WIZUALIZACJA 5: Zmienne kategoryczne vs score
if len(cat_features) > 0:
    n_cat = len(cat_features)
    n_cols_cat = 2
    n_rows_cat = (n_cat + n_cols_cat - 1) // n_cols_cat
    
    fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(14, n_rows_cat * 4))
    axes = axes.flatten() if n_cat > 1 else [axes]
    
    for idx, col in enumerate(cat_features):
        data.boxplot(column='score', by=col, ax=axes[idx])
        axes[idx].set_title(f'Score vs {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Score')
    
    for idx in range(n_cat, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('05_zmienne_kategoryczne_vs_score.png', dpi=300, bbox_inches='tight')
    print("\n✅ Wykres zapisany: 05_zmienne_kategoryczne_vs_score.png")
    plt.close()

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Podział danych:")
print(f" - Zbiór treningowy: {len(X_train)} próbek ({len(X_train)/len(data)*100:.1f}%)")
print(f" - Zbiór testowy: {len(X_test)} próbek ({len(X_test)/len(data)*100:.1f}%)")

# Pipeline przetwarzania
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ]
)

print("\n✅ Pipeline przetwarzania danych utworzony:")
print("   - Standaryzacja zmiennych numerycznych (StandardScaler)")
print("   - One-Hot Encoding zmiennych kategorycznych")

# ===============================
# 3️⃣ WYBÓR I TRENOWANIE MODELU (5 pkt)
# ===============================
print("\n" + "=" * 60)
print("🤖 ETAP 3: WYBÓR I TRENOWANIE MODELI")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

print("\n📋 Wybrane modele do porównania:")
print("1. Linear Regression - klasyczna regresja liniowa")
print("2. Ridge Regression - regresja z regularyzacją L2")
print("3. Random Forest - ensemble drzew decyzyjnych")

# Walidacja danych
if X_train.isnull().any().any():
    X_train = X_train.fillna(0)
    print("\n🔧 Uzupełniono NaN w zbiorze treningowym")

if X_test.isnull().any().any():
    X_test = X_test.fillna(0)
    print("🔧 Uzupełniono NaN w zbiorze testowym")

results = []
trained_models = {}

print("\n⚙️ Trenowanie modeli...\n")

for name, model in models.items():
    print(f"➡️  Trening: {name}...")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline
    
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    
    results.append({
        "Model": name,
        "R² (train)": r2_train,
        "R² (test)": r2_test,
        "R² (CV)": cv_scores.mean(),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    })
    
    print(f"   R² (train): {r2_train:.4f}")
    print(f"   R² (test):  {r2_test:.4f}")
    print(f"   R² (CV):    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   MAE:        {mae:.4f}")
    print(f"   RMSE:       {rmse:.4f}\n")

# ===============================
# 4️⃣ OCENA I OPTYMALIZACJA MODELU (5 pkt)
# ===============================
print("=" * 60)
print("📊 ETAP 4: OCENA I OPTYMALIZACJA MODELI")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\n📋 PODSUMOWANIE WYNIKÓW:\n")
print(results_df.to_string(index=False))

best_model_name = results_df.loc[results_df["R² (test)"].idxmax(), "Model"]
best_model = trained_models[best_model_name]

print(f"\n🏆 Najlepszy model: {best_model_name}")
print(f"   R² (test) = {results_df.loc[results_df['Model'] == best_model_name, 'R² (test)'].values[0]:.4f}")

# WIZUALIZACJA 6: Porównanie modeli
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ['R² (test)', 'MAE', 'RMSE', 'R² (CV)']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    colors_bar = ['gold' if m == best_model_name else 'steelblue' for m in results_df['Model']]
    ax.barh(results_df['Model'], results_df[metric], color=colors_bar, edgecolor='black')
    ax.set_xlabel(metric, fontweight='bold')
    ax.set_title(f'Porównanie: {metric}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(results_df[metric]):
        ax.text(v, i, f' {v:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('06_porownanie_modeli.png', dpi=300, bbox_inches='tight')
print("\n✅ Wykres zapisany: 06_porownanie_modeli.png")
plt.close()

# WIZUALIZACJA 7: Rzeczywiste vs Przewidywane
y_pred_best = best_model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(y_test, y_pred_best, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Idealna predykcja')
axes[0].set_xlabel('Rzeczywiste wartości (y_test)', fontweight='bold')
axes[0].set_ylabel('Przewidywane wartości (y_pred)', fontweight='bold')
axes[0].set_title(f'Rzeczywiste vs Przewidywane - {best_model_name}', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

residuals = y_test - y_pred_best
axes[1].scatter(y_pred_best, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
axes[1].axhline(0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel('Przewidywane wartości', fontweight='bold')
axes[1].set_ylabel('Residua (błędy)', fontweight='bold')
axes[1].set_title('Analiza residuów', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('07_rzeczywiste_vs_przewidywane.png', dpi=300, bbox_inches='tight')
print("✅ Wykres zapisany: 07_rzeczywiste_vs_przewidywane.png")
plt.close()

# WIZUALIZACJA 8: Rozkład błędów
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Residua (błędy predykcji)')
axes[0].set_ylabel('Częstość')
axes[0].set_title('Rozkład błędów predykcji', fontsize=12, fontweight='bold')
axes[0].axvline(0, color='red', linestyle='--', lw=2)

stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot (normalność residuów)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_analiza_bledow.png', dpi=300, bbox_inches='tight')
print("✅ Wykres zapisany: 08_analiza_bledow.png")
plt.close()

# OPTYMALIZACJA - Grid Search dla najlepszego modelu
print("\n🔧 OPTYMALIZACJA HIPERPARAMETRÓW...")

if best_model_name == "Random Forest":
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        best_model, param_grid, cv=5, scoring='r2', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Najlepsze parametry: {grid_search.best_params_}")
    print(f"✅ R² po optymalizacji: {grid_search.best_score_:.4f}")
    
    y_pred_optimized = grid_search.predict(X_test)
    r2_optimized = r2_score(y_test, y_pred_optimized)
    print(f"✅ R² na zbiorze testowym (optymalizowany): {r2_optimized:.4f}")

# Zapisanie wyników
results_df.to_csv("model_results.csv", index=False)
print("\n📁 Wyniki zapisano do: model_results.csv")

# WIZUALIZACJA 9: Feature Importance (dla Random Forest)
if best_model_name == "Random Forest":
    model_rf = best_model.named_steps['model']
    feature_names = num_features.copy()
    
    if len(cat_features) > 0:
        encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        cat_feature_names = encoder.get_feature_names_out(cat_features)
        feature_names.extend(cat_feature_names)
    
    importances = model_rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align='center', alpha=0.7, edgecolor='black')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Ważność cechy (Feature Importance)', fontweight='bold')
    plt.title('Top 15 najważniejszych cech - Random Forest', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('09_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Wykres zapisany: 09_feature_importance.png")
    plt.close()

print("\n" + "=" * 60)
print("✅ PROCES ZAKOŃCZONY SUKCESEM!")
print("=" * 60)
print("\n📊 Wygenerowane wykresy:")
print("   1. 01_rozklad_zmiennej_docelowej.png")
print("   2. 02_macierz_korelacji.png")
print("   3. 03_korelacja_ze_score.png")
print("   4. 04_rozklady_zmiennych.png")
print("   5. 05_zmienne_kategoryczne_vs_score.png")
print("   6. 06_porownanie_modeli.png")
print("   7. 07_rzeczywiste_vs_przewidywane.png")
print("   8. 08_analiza_bledow.png")
print("   9. 09_feature_importance.png")
print("\n📄 Plik z wynikami: model_results.csv")
print("\n🎓 Wszystkie wykresy gotowe do dokumentacji!\n")