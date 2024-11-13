import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import catboost as cb
import pickle
import os

# Load the dataset
file_path = r'data1.csv'  # Update this path as needed
df = pd.read_csv(file_path)

# Clean 'Price in Rupees' column by removing commas and handling non-numeric values
df['Price in Rupees'] = df['Price in Rupees'].replace({',': ''}, regex=True)
df['Price in Rupees'] = pd.to_numeric(df['Price in Rupees'], errors='coerce')
df['Price in Rupees'].fillna(df['Price in Rupees'].median(), inplace=True)

# Drop unnecessary columns
df = df.drop(columns=['ImageUrl', 'Name'])

# Label Encoding for categorical columns
label_columns = ['Processor', 'Camera_details', 'Storage_details', 'Screen_size', 'Battery_details']
label_encoders = {}
for column in label_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['Price in Rupees'])
y = df['Price in Rupees']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
    "XGBoost": xgb.XGBRegressor(random_state=42, n_estimators=100),
    "CatBoost": cb.CatBoostRegressor(verbose=0, random_seed=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "DecisionTree": DecisionTreeRegressor(random_state=42)
}

# Fit and evaluate models
best_model = None
best_score = -float('inf')

print("\nModel Performance on Test Set:")
for name, model in models.items():
    model.fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)
    print(f"{name} R^2 Score: {r2_score:.4f}")
    
    # Check for the best model
    if r2_score > best_score:
        best_score = r2_score
        best_model = model

# Voting Regressor using top-performing models
voting_regressor = VotingRegressor(estimators=[
    ('RandomForest', models["RandomForest"]),
    ('GradientBoosting', models["GradientBoosting"]),
    ('XGBoost', models["XGBoost"]),
    ('CatBoost', models["CatBoost"])
])
voting_regressor.fit(X_train, y_train)
voting_r2_score = voting_regressor.score(X_test, y_test)
print(f"Ensemble Voting Regressor R^2 Score: {voting_r2_score:.4f}")

# Choose best model between individual models and Voting Regressor
if voting_r2_score > best_score:
    best_model = voting_regressor
    best_score = voting_r2_score

# Ensure the directory exists to save the models
# os.makedirs('models', exist_ok=True)

# Save the best model, scaler, label encoders, and feature columns using pickle
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoders.pkl', 'wb') as encoders_file:
    pickle.dump(label_encoders, encoders_file)

# Save the feature names in the same order used for training
with open('feature_columns.pkl', 'wb') as feature_columns_file:
    pickle.dump(X.columns.tolist(), feature_columns_file)

print("\nBest model, scaler, label encoders, and feature columns have been saved successfully!")
