import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF

# 1. Wczytanie danych
df = pd.read_csv('CollegeDistance.csv')

# 2. Eksploracja danych
exploration_file = 'exploration.png'
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Macierz korelacji')
plt.savefig(exploration_file)
plt.close()

# 3. Obsługa brakujących wartości
df = df.fillna(df.mean())

# 4. Podział na cechy i zmienną docelową
X = df.drop('score', axis=1)
y = df['score']

# 5. Identyfikacja zmiennych numerycznych i kategorycznych
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# 6. Przygotowanie pipeline
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# 7. Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Wybór modeli
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "MLPRegressor": MLPRegressor(random_state=42, max_iter=500)
}

# 9. Trenowanie i ocena modeli
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    results[name] = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred)
    }

# 10. Generowanie PDF z wynikami
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Raport Modelu Predykcyjnego", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", '', 12)
pdf.cell(0, 10, "Wyniki modeli:", ln=True)
for name, metrics in results.items():
    pdf.cell(0, 10, f"{name} - R2: {metrics['R2']:.3f}, MAE: {metrics['MAE']:.3f}, MSE: {metrics['MSE']:.3f}", ln=True)

pdf.image(exploration_file, x=10, y=80, w=180)
pdf.output("docs/README.pdf")
