from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load best model
best_model = load_model(r'C:\Users\dell\Downloads\senu\best_model.keras')

# Validation data generator
img_rows, img_cols = 48, 48
batch_size = 32
validation_data_dir = r'C:\Users\dell\Downloads\senu\test'

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
val_loss, val_accuracy = best_model.evaluate(validation_generator, steps=len(validation_generator), verbose=1)
print(f"\n✅ Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"📉 Validation Loss: {val_loss:.4f}")

# Optional: Detailed classification report
y_true = validation_generator.classes
y_pred_probs = best_model.predict(validation_generator, steps=len(validation_generator), verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys()))









