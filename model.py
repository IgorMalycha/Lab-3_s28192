import pandas as pd
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

# 2. Boolean -> 0/1
bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    df[col] = df[col].map({'True':1, 'False':0})

# 3. Eksploracja danych - tylko kolumny numeryczne
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Macierz korelacji')
plt.savefig('exploration.png')
plt.close()

# 4. Obsługa braków
df = df.fillna(df.mean())

# 5. Podział cechy/zmienna docelowa
X = df.drop('score', axis=1)
y = df['score']

# 6. Identyfikacja kolumn numerycznych i kategorycznych
num_features = X.select_dtypes(include=['int64','float64']).columns
cat_features = X.select_dtypes(include=['object']).columns  # gender, ethnicity, income, region

# 7. Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# 8. Podział na train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Modele i parametry do GridSearch
models_params = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {}
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "regressor__n_estimators": [50, 100],
            "regressor__max_depth": [None, 5],
            "regressor__min_samples_split": [2, 5]
        }
    },
    "MLPRegressor": {
        "model": MLPRegressor(random_state=42, max_iter=500),
        "params": {
            "regressor__hidden_layer_sizes": [(50,), (100,)],
            "regressor__activation": ["relu", "tanh"],
            "regressor__alpha": [0.0001, 0.001]
        }
    }
}

results = {}
best_models = {}

for name, mp in models_params.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', mp['model'])
    ])
    
    if mp['params']:
        grid = GridSearchCV(pipeline, mp['params'], cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)  # <- pipeline koduje kategorie na liczby
        best_model = grid.best_estimator_
        results[name] = {
            "R2": r2_score(y_test, best_model.predict(X_test)),
            "MAE": mean_absolute_error(y_test, best_model.predict(X_test)),
            "MSE": mean_squared_error(y_test, best_model.predict(X_test)),
            "BestParams": grid.best_params_
        }
        best_models[name] = best_model
    else:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        results[name] = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "BestParams": None
        }
        best_models[name] = pipeline

# 10. Generowanie PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Raport Modelu Predykcyjnego", ln=True, align='C')
pdf.ln(10)

pdf.set_font("Arial", '', 12)
pdf.cell(0, 10, "Wyniki modeli:", ln=True)
for name, metrics in results.items():
    pdf.cell(0, 10, f"{name} - R2: {metrics['R2']:.3f}, MAE: {metrics['MAE']:.3f}, MSE: {metrics['MSE']:.3f}", ln=True)
    if metrics["BestParams"]:
        pdf.multi_cell(0, 10, f"  Najlepsze parametry: {metrics['BestParams']}")

pdf.image('exploration.png', x=10, y=130, w=180)
pdf.output("docs/README.pdf")
