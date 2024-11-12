import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# Load the dataset
file_path = r'E:\SEM V\ML CA2\data1.csv'  # Update this path as needed
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

# Train the RandomForestRegressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Ensure the directory exists to save the models
os.makedirs('models', exist_ok=True)

# Save the model, scaler, label encoders, and feature columns using pickle
with open('models/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('models/label_encoders.pkl', 'wb') as encoders_file:
    pickle.dump(label_encoders, encoders_file)

# Save the feature names in the same order used for training
with open('models/feature_columns.pkl', 'wb') as feature_columns_file:
    pickle.dump(X.columns.tolist(), feature_columns_file)

print("Model, scaler, label encoders, and feature columns have been saved successfully!")
