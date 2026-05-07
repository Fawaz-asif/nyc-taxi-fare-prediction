"""
NYC Taxi Fare Prediction - Training Script
============================================
Uses Artificial Neural Network (MLPRegressor) from scikit-learn
Dataset: NYC Taxi Fare Prediction (2020 Yellow Taxi Trip Records)

Student: M FAWAZ ASIF
Reg No: B23F0115CS070
University: Pak Austria Fachhochschule
"""

import pandas as pd
import numpy as np
import warnings
import os
import joblib
import json
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── 1. LOAD DATA ───────────────────────────────────────────────────────────────
print("=" * 70)
print("  NYC TAXI FARE PREDICTION - ANN TRAINING PIPELINE")
print("=" * 70)

print("\n[1/7] Loading dataset...")
start = time.time()

# The dataset is ~566MB / 6.4M rows. We'll sample for efficient training.
# Read in chunks and sample
chunk_size = 500_000
sampled_chunks = []
total_rows = 0

for chunk in pd.read_csv("data.csv", chunksize=chunk_size, low_memory=False):
    total_rows += len(chunk)
    # Sample 15% from each chunk for a representative sample
    sampled_chunks.append(chunk.sample(frac=0.15, random_state=42))

df = pd.concat(sampled_chunks, ignore_index=True)
print(f"  Total rows in dataset: {total_rows:,}")
print(f"  Sampled rows for training: {len(df):,}")
print(f"  Time: {time.time()-start:.1f}s")

# ─── 2. DATA EXPLORATION ────────────────────────────────────────────────────────
print("\n[2/7] Exploring data...")
print(f"\n  Columns ({len(df.columns)}):")
for col in df.columns:
    print(f"    - {col}: {df[col].dtype} | nulls: {df[col].isnull().sum()} | sample: {df[col].iloc[0]}")

print(f"\n  Target (fare_amount) statistics:")
print(f"    Mean: ${df['fare_amount'].mean():.2f}")
print(f"    Median: ${df['fare_amount'].median():.2f}")
print(f"    Std: ${df['fare_amount'].std():.2f}")
print(f"    Min: ${df['fare_amount'].min():.2f}")
print(f"    Max: ${df['fare_amount'].max():.2f}")

# ─── 3. DATA PREPROCESSING ──────────────────────────────────────────────────────
print("\n[3/7] Preprocessing data...")

# Convert datetime columns
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')

# Drop rows with null datetimes
df = df.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

# Remove invalid/outlier data
print("  Cleaning outliers and invalid data...")
original_len = len(df)

# Remove negative fares and extreme outliers
df = df[df['fare_amount'] > 0]
df = df[df['fare_amount'] <= 200]  # Cap at $200

# Remove invalid trip distances
df = df[df['trip_distance'] >= 0]
df = df[df['trip_distance'] <= 100]  # Cap at 100 miles

# Remove invalid passenger counts
df = df[df['passenger_count'] > 0]
df = df[df['passenger_count'] <= 6]

# Remove invalid rate codes (1-6 are valid)
df = df[df['RatecodeID'].isin([1, 2, 3, 4, 5, 6])]

# Remove invalid payment types (1-4 are valid)
df = df[df['payment_type'].isin([1, 2, 3, 4])]

# Remove invalid vendor IDs
df = df[df['VendorID'].isin([1, 2])]

print(f"  Rows removed: {original_len - len(df):,} ({(original_len - len(df))/original_len*100:.1f}%)")
print(f"  Remaining rows: {len(df):,}")

# ─── 4. FEATURE ENGINEERING ─────────────────────────────────────────────────────
print("\n[4/7] Engineering features...")

# Time-based features
df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
df['is_weekend'] = (df['pickup_day_of_week'] >= 5).astype(int)

# Rush hour flags
df['is_morning_rush'] = ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9)).astype(int)
df['is_evening_rush'] = ((df['pickup_hour'] >= 16) & (df['pickup_hour'] <= 19)).astype(int)
df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

# Night ride flag
df['is_night'] = ((df['pickup_hour'] >= 20) | (df['pickup_hour'] <= 5)).astype(int)

# Trip duration in minutes
df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0

# Remove invalid durations
df = df[df['trip_duration_min'] > 0]
df = df[df['trip_duration_min'] <= 180]  # Cap at 3 hours

# Speed (mph)
df['avg_speed_mph'] = np.where(
    df['trip_duration_min'] > 0,
    df['trip_distance'] / (df['trip_duration_min'] / 60.0),
    0
)
# Cap speed at reasonable limits
df = df[df['avg_speed_mph'] <= 80]

# Fare per mile
df['fare_per_mile'] = np.where(
    df['trip_distance'] > 0,
    df['fare_amount'] / df['trip_distance'],
    0
)

# Store and forward flag encoding
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).fillna(0)

# Fill any remaining NaN values
df = df.fillna(0)

# ─── 5. FEATURE SELECTION & PREPARATION ──────────────────────────────────────────
print("\n[5/7] Preparing features for training...")

feature_columns = [
    'VendorID',
    'passenger_count',
    'trip_distance',
    'RatecodeID',
    'store_and_fwd_flag',
    'PULocationID',
    'DOLocationID',
    'payment_type',
    'extra',
    'mta_tax',
    'tip_amount',
    'tolls_amount',
    'improvement_surcharge',
    'congestion_surcharge',
    'pickup_hour',
    'pickup_day_of_week',
    'pickup_month',
    'is_weekend',
    'is_rush_hour',
    'is_night',
    'trip_duration_min',
    'avg_speed_mph',
]

target_column = 'fare_amount'

X = df[feature_columns].values
y = df[target_column].values

print(f"  Feature matrix shape: {X.shape}")
print(f"  Target vector shape: {y.shape}")
print(f"  Features used ({len(feature_columns)}):")
for f in feature_columns:
    print(f"    - {f}")

# ─── 6. TRAIN-TEST SPLIT & SCALING ──────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n  Train set: {X_train.shape[0]:,} samples")
print(f"  Test set:  {X_test.shape[0]:,} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ─── 7. TRAIN ANN MODEL ─────────────────────────────────────────────────────────
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n[6/7] Training Artificial Neural Network (MLPRegressor)...")
print("  Architecture: 128 → 64 → 32 neurons")
print("  Activation: ReLU | Solver: Adam | Max iterations: 200")
print("  This may take a few minutes...")

start = time.time()

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    batch_size=1024,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    verbose=True,
)

model.fit(X_train_scaled, y_train)
training_time = time.time() - start

print(f"\n  Training completed in {training_time:.1f}s")
print(f"  Iterations: {model.n_iter_}")
print(f"  Best validation score: {model.best_validation_score_:.4f}")

# ─── 8. EVALUATION ──────────────────────────────────────────────────────────────
print("\n[7/7] Evaluating model...")

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("\n" + "=" * 50)
print("  MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"  {'Metric':<25} {'Train':>10} {'Test':>10}")
print(f"  {'-'*45}")
print(f"  {'MAE ($)':<25} {train_mae:>10.3f} {test_mae:>10.3f}")
print(f"  {'RMSE ($)':<25} {train_rmse:>10.3f} {test_rmse:>10.3f}")
print(f"  {'R² Score':<25} {train_r2:>10.4f} {test_r2:>10.4f}")
print(f"  {'MAPE (%)':<25} {train_mape:>10.2f} {test_mape:>10.2f}")
print("=" * 50)

# ─── 9. SAVE ARTIFACTS ──────────────────────────────────────────────────────────
print("\n  Saving model and artifacts...")

# Save model
joblib.dump(model, 'ann_model.joblib')
print(f"  ✓ Model saved: ann_model.joblib ({os.path.getsize('ann_model.joblib')/1024/1024:.1f} MB)")

# Save scaler
joblib.dump(scaler, 'scaler.joblib')
print(f"  ✓ Scaler saved: scaler.joblib")

# Save feature columns
joblib.dump(feature_columns, 'feature_columns.joblib')
print(f"  ✓ Feature columns saved: feature_columns.joblib")

# Save metrics
metrics = {
    'model': 'MLPRegressor (Artificial Neural Network)',
    'architecture': '128 → 64 → 32',
    'activation': 'relu',
    'solver': 'adam',
    'training_samples': int(X_train.shape[0]),
    'test_samples': int(X_test.shape[0]),
    'total_features': int(len(feature_columns)),
    'feature_names': feature_columns,
    'training_time_seconds': round(training_time, 1),
    'iterations': int(model.n_iter_),
    'metrics': {
        'train_mae': round(train_mae, 4),
        'test_mae': round(test_mae, 4),
        'train_rmse': round(train_rmse, 4),
        'test_rmse': round(test_rmse, 4),
        'train_r2': round(train_r2, 4),
        'test_r2': round(test_r2, 4),
        'train_mape': round(train_mape, 2),
        'test_mape': round(test_mape, 2),
    },
    'timestamp': datetime.now().isoformat(),
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ Metrics saved: metrics.json")

# Save loss curve data
loss_curve = model.loss_curve_
with open('loss_curve.json', 'w') as f:
    json.dump(loss_curve, f)
print(f"  ✓ Loss curve saved: loss_curve.json")

# ─── 10. GENERATE PLOTS ─────────────────────────────────────────────────────────
print("\n  Generating evaluation plots...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NYC Taxi Fare Prediction - ANN Model Evaluation', fontsize=16, fontweight='bold')

    # Plot 1: Training Loss Curve
    axes[0, 0].plot(model.loss_curve_, color='#e74c3c', linewidth=2)
    axes[0, 0].set_title('Training Loss Curve', fontweight='bold')
    axes[0, 0].set_xlabel('Iterations')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Actual vs Predicted (test set, sample)
    sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
    axes[0, 1].scatter(y_test[sample_idx], y_pred_test[sample_idx], alpha=0.3, s=5, color='#3498db')
    axes[0, 1].plot([0, 200], [0, 200], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 1].set_title('Actual vs Predicted Fare', fontweight='bold')
    axes[0, 1].set_xlabel('Actual Fare ($)')
    axes[0, 1].set_ylabel('Predicted Fare ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 100)
    axes[0, 1].set_ylim(0, 100)

    # Plot 3: Residual Distribution
    residuals = y_test - y_pred_test
    axes[1, 0].hist(residuals, bins=100, color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Residual Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Residual ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xlim(-30, 30)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Metrics comparison bar chart
    metrics_names = ['MAE', 'RMSE', 'R²']
    train_vals = [train_mae, train_rmse, train_r2]
    test_vals = [test_mae, test_rmse, test_r2]
    x = np.arange(len(metrics_names))
    width = 0.35
    bars1 = axes[1, 1].bar(x - width/2, train_vals, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = axes[1, 1].bar(x + width/2, test_vals, width, label='Test', color='#e74c3c', alpha=0.8)
    axes[1, 1].set_title('Model Metrics Comparison', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plots saved: evaluation_plots.png")

except ImportError:
    print("  ⚠ matplotlib not available, skipping plots")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE! All artifacts saved.")
print("=" * 70)
