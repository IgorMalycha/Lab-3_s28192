import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========================
# Wczytywanie danych
# ========================
print("="*70)
print("### Wczytywanie danych")
print("="*70)

data_path = "CollegeDistance.csv"
df = pd.read_csv(data_path)

print(f"Ścieżka: {data_path}")
print(f"Liczba rekordów: {df.shape[0]}")
print(f"Liczba kolumn: {df.shape[1]}")
print(f"Nazwy kolumn: {list(df.columns)}\n")
print("Podgląd (5 wierszy):")
print(df.head(), "\n")

# ========================
# Eksploracja danych
# ========================
print("="*70)
print("### Eksploracja danych (braki, typy, statystyki)")
print("="*70)

# Liczenie braków
total_cells = np.prod(df.shape)
missing_count = df.isnull().sum().sum()
missing_percent = missing_count / total_cells * 100
print(f"Brakujące przed czyszczeniem: {missing_count} ({missing_percent:.3f} % komórek)\n")

# Braki wg kolumn
print("Braki wg kolumn (przed czyszczeniem):")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Mapowanie 'yes'/'no' na 1/0 w kolumnach logicznych
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# Sprawdzenie braków po mapowaniu
missing_count_after = df.isnull().sum().sum()
print(f"\nBraki po wstępnej konwersji bool i konwersji numerycznej: {missing_count_after}")

# Typy zmiennych
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nRozpoznane zmienne numeryczne ({len(numeric_cols)}): {numeric_cols}")
print(f"Rozpoznane zmienne kategoryczne ({len(categorical_cols)}): {categorical_cols}")

print("="*70)
print("### Statystyki opisowe (data.describe())")
print("="*70)
print(df.describe(), "\n")

# ========================
# Generowanie wykresów eksploracyjnych
# ========================
print("="*70)
print("### Generowanie wykresów eksploracyjnych (zapisywane do plików)")
print("="*70)

plt.figure(figsize=(8,6))
sns.histplot(df['score'], bins=30, kde=True)
plt.title("Rozkład zmiennej score")
plt.savefig("score_distribution.png")
plt.close()
print("Zapisano: ./score_distribution.png")

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mapa korelacji")
plt.savefig("correlation_heatmap.png")
plt.close()
print("Zapisano: ./correlation_heatmap.png")

plt.figure(figsize=(8,6))
sns.boxplot(x='income', y='score', data=df)
plt.title("Score w zależności od income")
plt.savefig("boxplot_score_by_income.png")
plt.close()
print("Zapisano: ./boxplot_score_by_income.png\n")

# ========================
# Inżynieria cech
# ========================
print("="*70)
print("### Inżynieria cech i przygotowanie danych")
print("="*70)

# Usuwamy kolumnę 'rownames'
df = df.drop(columns=['rownames'])
print("Usunięto kolumnę 'rownames' (identyfikator).")

# Tworzymy nową cechę
df['wage_x_distance'] = df['wage'] * df['distance']
print("Dodano cechę 'wage_x_distance' (wage * distance).")

# Aktualizacja list zmiennych
numeric_cols = ['fcollege', 'mcollege', 'home', 'urban', 'unemp', 'wage', 'distance', 'tuition', 'education', 'wage_x_distance']
categorical_cols = ['gender', 'ethnicity', 'income', 'region']

print(f"Finalne zmienne numeryczne ({len(numeric_cols)}): {numeric_cols}")
print(f"Finalne zmienne kategoryczne ({len(categorical_cols)}): {categorical_cols}\n")

# ========================
# Podział danych
# ========================
print("="*70)
print("### Podział danych")
print("="*70)

X = df.drop(columns=['score'])
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Rozmiar zestawu treningowego: {X_train.shape} (≈{len(X_train)/len(df)*100:.1f}%)")
print(f"Rozmiar zestawu testowego:     {X_test.shape} (≈{len(X_test)/len(df)*100:.1f}%)\n")

# ========================
# Pipeline dla preprocessing
# ========================
print("="*70)
print("### Budowa pipeline'u (imputacja, skalowanie, encoding)")
print("="*70)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# ========================
# Modele do trenowania
# ========================
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Ridge': Ridge(alpha=1.0)
}

results = []

for name, model in models.items():
    print(f"\n➡️  Trening modelu: {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R²: {r2:.4f} | MAE: {mae:.4f} | MSE: {mse:.4f}")
    results.append({'model': name, 'R2': r2, 'MAE': mae, 'MSE': mse})

# ========================
# Podsumowanie wyników
# ========================
print("\n" + "="*70)
print("### Podsumowanie wyników modeli")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("model_results.csv", index=False)
print("\nWyniki zapisano do: model_results.csv")
