import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import PartialDependenceDisplay
import warnings

warnings.filterwarnings("ignore")

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

data = pd.read_csv("CollegeDistance.csv")

if 'rownames' in data.columns:
    data.drop(columns=['rownames'], inplace=True)

data = data.replace([" ", "NA", "N/A", "na", "NaN", "None"], np.nan)

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

for col in data.select_dtypes(include=[np.number]).columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

for col in data.select_dtypes(include=["object"]).columns:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

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
plt.close()

numeric_cols = data.select_dtypes(include=[np.number]).columns
corr = data[numeric_cols].corr()['score'].abs().sort_values(ascending=False)
top_features = corr.index[1:6].tolist()

try:
    sns.pairplot(data, vars=top_features + ['score'], kind='reg', plot_kws={'line_kws':{'color':'red','alpha':0.6}})
    plt.suptitle('Pairplot: top cech vs score', y=1.02)
    plt.savefig('EX_pairplot_top_features.png', dpi=300, bbox_inches='tight')
    plt.close()
except:
    pass

cat_features = data.select_dtypes(include=['object']).columns.tolist()
if len(cat_features) == 0:
    low_cardinality = [c for c in numeric_cols if data[c].nunique() < 10 and c != 'score']
    cat_features = low_cardinality

if len(cat_features) > 0:
    n = len(cat_features)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(cat_features):
        sns.countplot(x=col, data=data, ax=axes[i])
        axes[i].set_title(f'Rozkład: {col}')
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig('EX_countplots_categorical.png', dpi=300, bbox_inches='tight')
    plt.close()

try:
    sns.clustermap(data[numeric_cols].corr(), cmap='coolwarm', figsize=(10, 10), annot=True)
    plt.suptitle('Clustermap korelacji zmiennych numerycznych', y=1.02)
    plt.savefig('EX_clustermap_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
except:
    pass

target = 'score'
X = data.drop(columns=[target])
y = data[target]
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, num_features), ('cat', categorical_transformer, cat_features)])

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

trained_models = {}
results = []

for name, model in models.items():
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
results_df.to_csv('model_results.csv', index=False)

best_model_name = results_df.loc[results_df['R² (test)'].idxmax(), 'Model']
best_pipeline = trained_models[best_model_name]

train_sizes, train_scores, test_scores = learning_curve(best_pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Train score')
plt.plot(train_sizes, test_scores_mean, 'o-', label='CV score')
plt.xlabel('Liczba próbek treningowych')
plt.ylabel('R²')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('EX_learning_curve_best_model.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(len(models), 2, figsize=(12, 4*len(models)))
for i, (name, pipeline) in enumerate(trained_models.items()):
    y_pred = pipeline.predict(X_test)
    resid = y_test - y_pred
    axes[i,0].hist(resid, bins=30, edgecolor='black', alpha=0.7)
    axes[i,0].set_xlabel('Residua')
    axes[i,0].grid(True, alpha=0.3)
    axes[i,1].boxplot(resid)
plt.tight_layout()
plt.savefig('EX_residuals_per_model.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test, alpha=0.6, label='Ideal')
for name, pipeline in trained_models.items():
    y_pred = pipeline.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.5, label=name)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
plt.xlabel('Rzeczywiste')
plt.ylabel('Przewidywane')
plt.legend()
plt.tight_layout()
plt.savefig('EX_real_vs_pred_all_models.png', dpi=300, bbox_inches='tight')
plt.close()

if 'Random Forest' in trained_models:
    rf = trained_models['Random Forest'].named_steps['model']
    feature_names = num_features.copy()
    if len(cat_features) > 0:
        encoder = trained_models['Random Forest'].named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        cat_names = encoder.get_feature_names_out(cat_features)
        feature_names.extend(cat_names)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(10,8))
    plt.barh(range(len(idx)), importances[idx], edgecolor='black', alpha=0.7)
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('EX_feature_importance_rf_top20.png', dpi=300, bbox_inches='tight')
    plt.close()

for lin_name in ['Linear Regression', 'Ridge Regression']:
    if lin_name in trained_models:
        model = trained_models[lin_name].named_steps['model']
        feature_names = num_features.copy()
        if len(cat_features) > 0:
            encoder = trained_models[lin_name].named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
            cat_names = encoder.get_feature_names_out(cat_features)
            feature_names.extend(cat_names)
        try:
            coefs = np.array(model.coef_).flatten()
            idx = np.argsort(np.abs(coefs))[::-1][:20]
            plt.figure(figsize=(10,8))
            plt.barh(range(len(idx)), coefs[idx], edgecolor='black', alpha=0.7)
            plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
            plt.gca().invert_yaxis()
            fname = f'EX_coefs_{lin_name.replace(" ","_")}.png'
            plt.tight_layout()
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        except:
            pass

try:
    top3 = top_features[:3]
    for feat in top3:
        plt.figure(figsize=(6,4))
        PartialDependenceDisplay.from_estimator(best_pipeline, X_test, [feat])
        plt.tight_layout()
        plt.savefig(f'EX_pdp_{feat}.png', dpi=300, bbox_inches='tight')
        plt.close()
except:
    pass

score_corr = data[numeric_cols].corr()['score'].sort_values()
plt.figure(figsize=(8,6))
colors = ['green' if v>0 else 'red' for v in score_corr.values]
plt.barh(score_corr.index, score_corr.values, color=colors, edgecolor='black')
plt.tight_layout()
plt.savefig('EX_corr_vs_score_sorted.png', dpi=300, bbox_inches='tight')
plt.close()
