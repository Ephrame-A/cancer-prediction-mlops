import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import shutil

# --- Configuration ---
MIN_ACCURACY = 0.90
BASE_MODEL_DIR = 'models'
MODEL_NAME = 'my_model'
DATA_PATH = './data/Lung Cancer Dataset.csv'  # Updated to match actual CSV file name

# --- Step 1: Data Preparation ---
print("Loading and preparing dataset...")
df = pd.read_csv(DATA_PATH)


# Assuming the last column is the target and the rest are features
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]
# Convert target to numeric if it contains YES/NO
if y.dtype == object or y.apply(lambda v: isinstance(v, str)).any():
    y = y.map({'YES': 1, 'NO': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 2: Model Training ---
print("Training new model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),  # Adjust input shape based on features
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30, validation_data=(X_test_scaled, y_test), verbose=0)

# --- Step 3: Accuracy Gate Check ---
print("Evaluating new model...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"New model accuracy: {accuracy * 100:.2f}%")

if accuracy >= MIN_ACCURACY:
    print("New model meets accuracy threshold. Proceeding with promotion.")

    # Create models folder if it doesn't exist
    if not os.path.exists(BASE_MODEL_DIR):
        os.makedirs(BASE_MODEL_DIR)

    # Find the next version number
    current_versions = [int(v) for v in os.listdir(BASE_MODEL_DIR) if v.isdigit() and os.path.isdir(os.path.join(BASE_MODEL_DIR, v))]
    next_version = max(current_versions) + 1 if current_versions else 1

    # Define the new model path
    export_path = os.path.join(BASE_MODEL_DIR, str(next_version))

    # Save the model using model.save() for TensorFlow
    try:
        model.export(export_path)
        print(f"Model successfully promoted and saved to: {export_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

else:
    print("New model failed accuracy check. Keeping the previous version.")