import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import json
import gc
import joblib
import os

# 1. Load Data
file_path = r"C:\Users\M S I\Downloads\cancer_dataset\jan to may police violation_anonymized791b166.csv"
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Data Preprocessing & Leakage Prevention
print("Preprocessing data and extracting features...")

# Parse violation_type from JSON array string to first element
def parse_violation(x):
    if pd.isna(x):
        return "UNKNOWN"
    try:
        # e.g., '["WRONG PARKING","PARKING NEAR ROAD CROSSING"]' -> 'WRONG PARKING'
        parsed = json.loads(x)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return "UNKNOWN"
    except:
        return str(x)

df['target_violation'] = df['violation_type'].apply(parse_violation)

# Drop rows with unknown target
df = df[df['target_violation'] != "UNKNOWN"].copy()

# Feature Engineering from created_datetime
df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
df = df.dropna(subset=['created_datetime'])

# Extract temporal features
df['hour_of_day'] = df['created_datetime'].dt.hour
df['day_of_week'] = df['created_datetime'].dt.dayofweek
df['month'] = df['created_datetime'].dt.month

# Keep relevant features, drop leaky/unnecessary ones
# vehicle_type, police_station have some NaNs, fill them
df['vehicle_type'] = df['vehicle_type'].fillna('UNKNOWN')
df['police_station'] = df['police_station'].fillna('UNKNOWN')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Drop rows with invalid coordinates
df = df.dropna(subset=['latitude', 'longitude'])

features = ['latitude', 'longitude', 'vehicle_type', 'police_station', 'hour_of_day', 'day_of_week']
target = 'target_violation'

X = df[features]
y = df[target]

# Keep top N classes to make the report readable if there are too many rare ones
top_classes = y.value_counts().nlargest(10).index
mask = y.isin(top_classes)
X = X[mask]
y = y[mask]

# 3. Data Splitting (Chronological to prevent leakage)
print("Splitting data chronologically...")
df_sorted = df.loc[X.index].sort_values('created_datetime')
X_sorted = df_sorted[features]
y_sorted = df_sorted[target]

split_idx = int(len(X_sorted) * 0.8) # 80% train, 20% test (chronological)

X_train = X_sorted.iloc[:split_idx]
y_train = y_sorted.iloc[:split_idx]
X_test = X_sorted.iloc[split_idx:]
y_test = y_sorted.iloc[split_idx:]

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Free memory before model training
del df
del df_sorted
del X_sorted
del y_sorted
gc.collect()

# 4. Model Pipeline
print("Building and training pipeline...")
numeric_features = ['latitude', 'longitude', 'hour_of_day', 'day_of_week']
categorical_features = ['vehicle_type', 'police_station']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Random Forest with restricted depth to prevent overfitting
model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=2)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Train
pipeline.fit(X_train, y_train)

# 5. Evaluation
print("Evaluating model...")
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("-" * 50)
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")

# Check for overfitting
if train_acc - test_acc > 0.1:
    print("WARNING: Model may be overfitting! (Train Acc >> Test Acc)")
else:
    print("SUCCESS: Model shows minimal overfitting.")

print("-" * 50)
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, zero_division=0))

# 6. Save Model
model_save_path = os.path.join(os.path.dirname(__file__), "backend", "data", "random_forest_pipeline.joblib")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
print(f"Saving model pipeline to {model_save_path}...")
joblib.dump(pipeline, model_save_path)
print("Random Forest pipeline saved successfully!")
