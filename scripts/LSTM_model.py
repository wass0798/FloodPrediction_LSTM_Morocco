# ==========================================
# Flood Prediction LSTM Pipeline
# ==========================================

import pandas as pd
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. Load Data
# ==========================================

# Load sample points (Excel file should have 'lat', 'lon', 'Target' columns)
samples_df = pd.read_excel("your_path/sample_points.xlsx")

# Load raster data (multi-band)
raster_path = "your_path/raster.tif"
with rasterio.open(raster_path) as src:
    raster_data = src.read()
    print("Raster shape:", raster_data.shape)  # (bands, height, width)

# ==========================================
# 2. Data Extraction and Augmentation
# ==========================================

def patch_extraction(raster, samples_df):
    """Extract raster values at sample points for all bands."""
    sample_values = []
    for _, row in samples_df.iterrows():
        lon, lat = row['lon'], row['lat']
        row_col = src.index(lon, lat)  # Convert lat/lon to raster row/col
        pixel_values = raster[:, row_col[0], row_col[1]]
        sample_values.append(pixel_values)
    extracted_df = pd.DataFrame(sample_values)
    extracted_df['Target'] = samples_df['Target'].values
    return extracted_df

def spatial_perturbation(raster, samples_df):
    """Slightly perturb latitude and longitude to augment data."""
    perturbed_df = samples_df.copy()
    perturbed_df['lat'] += np.random.uniform(-0.001, 0.001, len(samples_df))
    perturbed_df['lon'] += np.random.uniform(-0.001, 0.001, len(samples_df))
    return patch_extraction(raster, perturbed_df)

def add_noise_to_raster_values(raster, samples_df):
    """Add Gaussian noise to raster data for augmentation."""
    noise = np.random.normal(0, 0.01, raster.shape)
    noisy_raster = raster + noise
    return patch_extraction(noisy_raster, samples_df)

def augment_data(raster, samples_df):
    """Combine all augmentation methods into a single dataset."""
    extraction = patch_extraction(raster, samples_df)
    perturbed = spatial_perturbation(raster, samples_df)
    noisy = add_noise_to_raster_values(raster, samples_df)

    # Combine and remove duplicates
    combined = pd.concat([extraction, perturbed, noisy], ignore_index=True).drop_duplicates()
    return combined

# Apply augmentation
augmented_dataset = augment_data(raster_data, samples_df)

# Rename columns to meaningful feature names
new_column_names = ["LU", "SPI", "DD", "DR", "NDVI", "Slope", "Elevation",
                    "TWI", "Curvature", "ASPECT", "TPI", "Target"]
augmented_dataset.columns = new_column_names

# Drop less relevant factors
df = augmented_dataset.drop(['Curvature', 'ASPECT'], axis=1)

# ==========================================
# 3. Prepare Data for LSTM
# ==========================================

X = df.drop('Target', axis=1)
Y = df['Target']

# Split into train/test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM (samples, timesteps=1, features)
X_train = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# ==========================================
# 4. Build and Train LSTM Model
# ==========================================

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.1),
    BatchNormalization(),
    LSTM(50),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=28,
    verbose=1,
    callbacks=[early_stopping]
)

# ==========================================
# 5. Evaluate Model
# ==========================================

# Predict and convert probabilities to binary labels
predictions = model.predict(X_test)
binary_predictions = (predictions >= 0.5).astype(int).reshape(-1)

# Confusion matrix plotting
def plot_confusion_matrix(y_true, y_pred):
    cm = pd.crosstab(y_true, y_pred)
    a = cm.to_numpy()
    labels_matrix = np.array([["TN", "FN", "FP", "TP"]]).flatten()
    labels_matrix = np.array(["{0}\n\n{1}".format(l, val) for l, val in zip(labels_matrix, a.flatten())]).reshape(2,2)
    sns.set(font_scale=1)
    sns.heatmap(cm, annot=labels_matrix, fmt='', cmap='Purples')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion_matrix(y_test, binary_predictions)

# Print key performance metrics
def print_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    re = np.abs((y_true - y_pred) / y_true).mean()

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"AUC: {auc*100:.2f}%")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Relative Error: {re:.4f}")

print_scores(y_test, binary_predictions)
