import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.preprocessing import load_data

# Load and prepare data
data_dir = "data/PlantVillage"  # Update if your dataset path is different
images, labels, class_names = load_data(data_dir)

# Normalize the images
images = images / 255.0

# Encode labels into numbers
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Load the trained model
model = load_model("models/crop_disease_model.h5")

# Make predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
