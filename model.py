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

# ===============================
# 1️⃣ Wczytanie danych
# ===============================
data = pd.read_csv("CollegeDistance.csv")

# ===============================
# 2️⃣ Poprawa typów i literówek
# ===============================
# Poprawna nazwa kolumny
categorical_features = ['gender', 'ethnicity', 'income', 'region']

# Konwersja bool na 0/1
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})

# Numeryczne kolumny do skalowania
numerical_features = ['unemp', 'wage', 'distance', 'tuition', 'education'] + bool_cols

# ===============================
# 3️⃣ Eksploracja danych
# ===============================
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Wykresy
sns.histplot(data['score'], kde=True)
plt.savefig("score_distribution.png")
plt.clf()

sns.boxplot(x='gender', y='score', data=data)
plt.savefig("score_by_gender.png")
plt.clf()

# ===============================
# 4️⃣ Podział danych
# ===============================
X = data.drop('score', axis=1)
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 5️⃣ Preprocessing
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# ===============================
# 6️⃣ Modele
# ===============================
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

best_model = None
best_score = -np.inf
best_model_name = ""

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R2: {r2:.3f}")
    if r2 > best_score:
        best_score = r2
        best_model = pipeline
        best_model_name = name

print(f"Najlepszy model: {best_model_name}")

# ===============================
# 7️⃣ Ocena najlepszego modelu
# ===============================
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)

# ===============================
# 8️⃣ Zapis modelu
# ===============================
joblib.dump(best_model, 'best_model.pkl')

# ===============================
# 9️⃣ Generowanie PDF
# ===============================
def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Tytuł
    pdf.cell(0, 10, "Analiza i Model Predykcyjny - Score", 0, 1, 'C')
    
    # Statystyki opisowe
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Statystyki opisowe zmiennych:", 0, 1)
    
    stats_text = data.describe().to_string()
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, stats_text)
    
    # Wykresy
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Wykresy eksploracyjne:", 0, 1)
    
    for img in ["score_distribution.png", "score_by_gender.png"]:
        pdf.image(img, w=180)
        pdf.ln(5)
    
    # Wyniki modelu
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Wyniki najlepszego modelu:", 0, 1)
    
    model_text = f"Najlepszy model: {best_model_name}\n"
    model_text += f"MAE: {mae:.3f}\n"
    model_text += f"MSE: {mse:.3f}\n"
    model_text += f"R2: {r2:.3f}"
    
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, model_text)
    
    pdf.output("README.pdf")

create_pdf()
