import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

# ✅ Custom Transformer to clean mileage, engine, max_power
class CleanNumericFields(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['mileage'] = X['mileage'].str.replace(' kmpl', '', regex=False)
        X['engine'] = X['engine'].str.replace(' CC', '', regex=False)
        X['max_power'] = X['max_power'].str.replace(' bhp', '', regex=False)

        X['mileage'] = pd.to_numeric(X['mileage'], errors='coerce')
        X['engine'] = pd.to_numeric(X['engine'], errors='coerce')
        X['max_power'] = pd.to_numeric(X['max_power'], errors='coerce')

        return X

# ✅ Load data
df = pd.read_csv('Car details v3.csv')
print("Columns in dataset:")
print(df.columns)

# ✅ Drop unused columns and separate features/target
df = df.dropna(subset=['selling_price'])

X = df.drop(columns=['selling_price', 'name'])  # features
y = df['selling_price']                         # target

# 🔧 Categorical and numeric columns
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

# 🔧 Build Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 🔧 Full pipeline: cleaning → preprocessing → model
full_pipeline = Pipeline(steps=[
    ('cleaner', CleanNumericFields()),
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Fit the pipeline
full_pipeline.fit(X_train, y_train)

# ✅ Predict and evaluate
y_pred = full_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ✅ Save the pipeline as a .pkl file
joblib.dump(full_pipeline, 'car_price_pipeline.pkl')