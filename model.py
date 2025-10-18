import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1️⃣ Wczytanie danych
data = pd.read_csv("CollegeDistance.csv")

# 2️⃣ Eksploracja danych
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Wykresy przykładowe
sns.histplot(data['score'], kde=True)
plt.savefig("score_distribution.png")
plt.clf()

sns.boxplot(x='gender', y='score', data=data)
plt.savefig("score_by_gender.png")
plt.clf()

# 3️⃣ Inżynieria cech
# Lista zmiennych kategorycznych i numerycznych
categorical_features = ['gender', 'etchnicity', 'income', 'region']
numerical_features = ['unemp', 'wage', 'distance', 'tuition', 'education',
                      'fcollege', 'mcollege', 'home', 'urban']

# One-hot encoding dla kategorii i skalowanie dla liczb
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# 4️⃣ Podział danych
X = data.drop('score', axis=1)
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ Wybór modelu (przykład: Random Forest, Gradient Boosting, Linear Regression)
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

best_model = None
best_score = -np.inf

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

print(f"Najlepszy model: {best_model}")

# 6️⃣ Zapis modelu
import joblib
joblib.dump(best_model, 'best_model.pkl')

# 7️⃣ Ocena modelu
y_pred = best_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
