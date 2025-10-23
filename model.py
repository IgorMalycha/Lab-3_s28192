import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

data = pd.read_csv("CollegeDistance.csv")
if 'rownames' in data.columns:
    data.drop(columns=['rownames'], inplace=True)

print(f"Liczba rekordów: {len(data)}")
print(f"Liczba kolumn: {len(data.columns)}")
print(f"Kolumny w datasecie: {list(data.columns)}\n")
print("Podgląd pierwszych 5 wierszy:")
print(data.head())

print(data.info())
print(data.describe())

missing_data = data.isnull().sum()
missing_percent = (missing_data / len(data)) * 100
missing_df = pd.DataFrame({
    'Kolumna': missing_data.index,
    'Braki': missing_data.values,
    'Procent': missing_percent.values
}).query('Braki > 0')

if len(missing_df) > 0:
    print("Brakujące wartości:")
    print(missing_df.to_string(index=False))
else:
    print("Brak brakujących wartości")

bool_cols = ['fcollege', 'mcollege', 'home', 'urban']
for col in bool_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).str.lower().map({"yes": 1, "no": 0})
        data[col].fillna(data[col].median(), inplace=True)

for col in data.select_dtypes(include=[np.number]).columns:
    data[col].fillna(data[col].median(), inplace=True)

for col in data.select_dtypes(include=["object"]).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

print(f"\nBrakujące wartości po czyszczeniu: {data.isnull().sum().sum()}\n")

target = 'score'
X = data.drop(columns=[target])
y = data[target]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

print("Zmienne numeryczne:", num_features)
print("Zmienne kategoryczne:", cat_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features),
    ('cat', categorical_transformer, cat_features)
])

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

trained_models = {}
results = []

for name, model in models.items():
    print(f"Trening: {name}")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
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
        'Model': name,
        'R² (train)': r2_train,
        'R² (test)': r2_test,
        'R² (CV)': cv_scores.mean(),
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    })

results_df = pd.DataFrame(results)
print("\nPodsumowanie wyników modeli:")
print(results_df)

results_df.to_csv('model_results.csv', index=False)

