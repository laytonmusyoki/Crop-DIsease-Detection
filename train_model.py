import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from src.preprocessing import load_data

# Load dataset
data_dir = "data/PlantVillage"  # update path if needed
images, labels, class_names = load_data(data_dir)

# Normalize pixel values
images = images.astype('float32') / 255.0

# Encode labels to numeric
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded, test_size=0.2, random_state=42
)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Define CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("models/best_model.h5", save_best_only=True)

# Train the model with class weights and callbacks
history = model.fit(
    X_train, y_train,
    epochs=25,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

# Save final model (optional, since best is saved separately)
model.save("models/crop_disease_model.h5")

# Save label encoder
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Optional: Plot training history
# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
