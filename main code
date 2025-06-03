from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import (Dense, Dropout, Flatten, BatchNormalization, 
Conv2D, MaxPooling2D, GaussianNoise, LeakyReLU) 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.losses import CategoricalCrossentropy 
from tensorflow.keras.regularizers import l2 
import matplotlib.pyplot as plt 
# Config 
num_classes = 5 
img_rows, img_cols = 48, 48 
batch_size = 32 
epochs = 150 
train_data_dir = r'C:\Users\dell\Downloads\senu\train' 
validation_data_dir = r'C:\Users\dell\Downloads\senu\test' 
# Data Augmentation 
train_datagen = ImageDataGenerator( 
rescale=1./255, 
rotation_range=20, 
shear_range=0.2, 
zoom_range=0.2, 
width_shift_range=0.25, 
height_shift_range=0.25, 
horizontal_flip=True, 
brightness_range=[0.8, 1.2], 
fill_mode='nearest' 
) 
validation_datagen = ImageDataGenerator(rescale=1./255) 
train_generator = train_datagen.flow_from_directory( 
train_data_dir, color_mode='grayscale', target_size=(img_rows, img_cols), 
batch_size=batch_size, class_mode='categorical', shuffle=True) 
validation_generator = validation_datagen.flow_from_directory( 
validation_data_dir, color_mode='grayscale', target_size=(img_rows, img_cols), 
batch_size=batch_size, class_mode='categorical', shuffle=False) 
# Model 
model = Sequential() 
# Block-1 
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', 
kernel_regularizer=l2(0.001), input_shape=(img_rows, img_cols, 1))) 
model.add(GaussianNoise(0.1)) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(BatchNormalization()) 
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.2)) 
# Block-2 
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(BatchNormalization()) 
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.3)) 
# Block-3 
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(BatchNormalization()) 
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.001))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(BatchNormalization()) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.3)) 
# Fully Connected 
model.add(Flatten()) 
model.add(Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(0.001))) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(BatchNormalization()) 
model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation='softmax')) 
model.summary() 
# Callbacks 
checkpoint = ModelCheckpoint(r'C:\Users\dell\Downloads\senu\best_model.keras', 
monitor='val_accuracy', mode='max', save_best_only=True, verbose=1) 
earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.002, patience=15, 
verbose=1, restore_best_weights=True) 
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=6, 
verbose=1, min_delta=0.001) 
callbacks = [earlystop, checkpoint, reduce_lr] 
# Compile 
loss_fn = CategoricalCrossentropy(label_smoothing=0.05) 
model.compile(loss=loss_fn, optimizer=Adam(learning_rate=0.0005), metrics=['accuracy']) 
# Train 
history = model.fit( 
train_generator, 
steps_per_epoch=len(train_generator), 
epochs=epochs, 
callbacks=callbacks, 
validation_data=validation_generator, 
validation_steps=len(validation_generator) 
) 
# Plot Accuracy 
plt.figure(figsize=(8, 6)) 
plt.plot(history.history['accuracy'], label='Train Accuracy') 
plt.plot(history.history['val_accuracy'], label='Val Accuracy') 
plt.title("Training vs Validation Accuracy") 
plt.xlabel("Epochs") 
plt.ylabel("Accuracy") 
plt.legend() 
plt.grid(True) 
plt.tight_layout() 
plt.show()
