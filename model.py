import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fpdf import FPDF
import joblib
import warnings
warnings.filterwarnings("ignore")

print("\n========================")
print("  ZADANIE 1 - MODEL PREDYKCYJNY")
print("========================\n")

# ===============================
# 1️⃣ Wczytanie danych
# ===============================
print("📂 Wczytywanie danych...")
data = pd.read_csv("CollegeDistance.csv")
print(f"Liczba rekordów: {data.shape[0]}")
print(f"Liczba kolumn: {data.shape[1]}\n")

# ===============================
# 2️⃣ Wstępna analiza danych
# ===============================
print("🔍 Analiza brakujących danych...")
missing_before = data.isnull().sum().sum()
print(f"Brakujące wartości przed czyszczeniem: {missing_before}")

# Jeśli są braki – imputacja medianą lub trybem
for col in data.columns:
    if data[col].isnull().any():
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)

missing_after = data.isnull().sum().sum()
print(f"Brakujące wartości po czyszczeniu: {missing_after}")

# Statystyka czyszczenia
if missing_before > 0:
    print(f"✅ Uzupełniono {(missing_before - missing_after) / len(data) * 100:.2f}% danych.\n")
else:
    print("✅ Brak brakujących danych.\n")

# ===============================
# 3️⃣ Konwersje typów i poprawki nazw
# ===============================
print("🧹 Przygotowanie danych...")

# Poprawna nazwa kolumny
categorical_features = ['gender', 'ethnicity', 'income', 'region']

# Konwersja kolumn bool na 0/1
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    data[col] = data[col].map({'True': 1, 'False': 0})

numerical_features = ['unemp', 'wage', 'distance', 'tuition', 'education'] + bool_cols

# Usuwanie kolumn niepotrzebnych
if "rownames" in data.columns:
    data = data.drop(columns=["rownames"])
    print("🗑️ Kolumna 'rownames' została usunięta.\n")

# ===============================
# 4️⃣ Eksploracja danych
# ===============================
print("📊 Tworzenie wykresów eksploracyjnych...")
sns.histplot(data['score'], kde=True)
plt.title("Rozkład zmiennej docelowej: score")
plt.savefig("score_distribution.png")
plt.clf()

sns.boxplot(x='gender', y='score', data=data)
plt.title("Wartość score w zależności od płci")
plt.savefig("score_by_gender.png")
plt.clf()

# ===============================
# 5️⃣ Podział danych
# ===============================
X = data.drop('score', axis=1)
y = data['score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📈 Podział danych:")
print(f" - Zbiór treningowy: {len(X_train)} próbek ({len(X_train)/len(data)*100:.1f}%)")
print(f" - Zbiór testowy: {len(X_test)} próbek ({len(X_test)/len(data)*100:.1f}%)\n")

# ===============================
# 6️⃣ Preprocessing
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# ===============================
# 7️⃣ Trenowanie i porównanie modeli
# ===============================
print("🤖 Trenowanie modeli...\n")
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

results = []
best_model = None
best_score = -np.inf
best_model_name = ""

for name, model in models.items():
    print(f"➡️  Trening modelu: {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mae, mse, rmse, r2))
    print(f"   MAE: {mae:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}\n")

    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_model_name = name

print("==============================")
print(f"🏆 Najlepszy model: {best_model_name} (R² = {best_score:.3f})")
print("==============================\n")

# ===============================
# 8️⃣ Ewaluacja i zapis modelu
# ===============================
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("📋 Końcowa ewaluacja najlepszego modelu:")
print(f" - MAE  (średni błąd bezwzględny): {mae:.4f}")
print(f" - MSE  (średni błąd kwadratowy): {mse:.4f}")
print(f" - RMSE (pierwiastek z MSE): {rmse:.4f}")
print(f" - R²   (współczynnik determinacji): {r2:.4f}\n")

# Zapis modelu
joblib.dump(best_model, 'best_model.pkl')
print("💾 Zapisano najlepszy model jako 'best_model.pkl'\n")

# ===============================
# 9️⃣ Generowanie PDF z raportem
# ===============================
print("📝 Generowanie raportu PDF...")

def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Analiza i Model Predykcyjny - Score", 0, 1, 'C')

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Statystyki opisowe:", 0, 1)
    stats_text = data.describe().to_string()
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, stats_text)

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Wykresy eksploracyjne:", 0, 1)
    for img in ["score_distribution.png", "score_by_gender.png"]:
        pdf.image(img, w=180)
        pdf.ln(5)

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Wyniki modeli:", 0, 1)
    pdf.set_font("Arial", "", 10)
    for name, mae, mse, rmse, r2 in results:
        pdf.multi_cell(0, 5, f"{name}\n  MAE: {mae:.3f}\n  MSE: {mse:.3f}\n  RMSE: {rmse:.3f}\n  R²: {r2:.3f}\n")

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Najlepszy model:", 0, 1)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, f"{best_model_name}\nMAE: {mae:.3f}\nMSE: {mse:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}")

    pdf.output("README.pdf")

create_pdf()
print("✅ Raport zapisany jako README.pdf\n")

print("==============================")
print("✅ ZAKOŃCZONO ANALIZĘ I UCZENIE MODELU")
print("==============================\n")
