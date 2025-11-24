import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, 
                                   BatchNormalization, Flatten, Dense, 
                                   GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import cv2
import os
import glob
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Libraries imported successfully!")

# Configuration and parameters
DATASET_PATH = r"C:\Users\gudur\Downloads\archive"
IMG_WIDTH = 48
IMG_HEIGHT = 48
CHANNELS = 1
NUM_CLASSES = 7
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

CLASS_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("Configuration set successfully!")
print(f"Dataset path: {DATASET_PATH}")
print(f"Classes: {CLASS_LABELS}")

# Dataset loading functions...

def load_dataset_from_train_test_folders(dataset_path):
    def load_images_from_folder(base_path, split_name):
        images = []
        labels = []
        print(f"\nLoading {split_name} data from {base_path}")
        for class_idx, class_name in enumerate(CLASS_LABELS):
            class_path = os.path.join(base_path, class_name)
            if os.path.exists(class_path):
                print(f"Loading {class_name} images from {class_path}")
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(glob.glob(os.path.join(class_path, ext)))
                for img_path in image_files:
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            if CHANNELS == 1:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            else:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                            if CHANNELS == 1:
                                img = np.expand_dims(img, axis=-1)
                            images.append(img)
                            labels.append(class_idx)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                class_count = len([l for l in labels if l == class_idx])
                print(f"  → Loaded {class_count} {class_name} images")
            else:
                print(f"Warning: {class_path} not found!")
        return np.array(images), np.array(labels)

    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    X_train, y_train = load_images_from_folder(train_path, 'training')
    X_test, y_test = load_images_from_folder(test_path, 'testing')

    return X_train, y_train, X_test, y_test

def preprocess_data(X, y):
    X = X.astype('float32') / 255.0
    y_categorical = to_categorical(y, num_classes=NUM_CLASSES)
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y_categorical.shape}")
    return X, y_categorical

def visualize_sample_images(X, y, class_labels, num_samples_per_class=1):
    fig, axes = plt.subplots(1, len(class_labels), figsize=(20, 3))
    for i, class_label in enumerate(class_labels):
        class_idx = np.where(y == i)[0]
        if len(class_idx) > 0:
            img_idx = class_idx[0]
            img = X[img_idx]
            if CHANNELS == 1:
                axes[i].imshow(img.squeeze(), cmap='gray')
            else:
                axes[i].imshow(img)
            axes[i].set_title(f'{class_label.title()}')
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axes[i].set_title(f'{class_label.title()}')
            axes[i].axis('off')
    plt.tight_layout()
    plt.show()

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset path '{DATASET_PATH}' not found!")
else:
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_dataset_from_train_test_folders(DATASET_PATH)
    print(f"\nDataset loaded successfully!")
    print(f"Training images: {X_train_raw.shape}")
    print(f"Training labels: {y_train_raw.shape}")
    print(f"Test images: {X_test_raw.shape}")
    print(f"Test labels: {y_test_raw.shape}")

print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

X_train, y_train_cat = preprocess_data(X_train_raw, y_train_raw)
X_test, y_test_cat = preprocess_data(X_test_raw, y_test_raw)

print("\nTraining Set Class Distribution:")
unique_train, counts_train = np.unique(y_train_raw, return_counts=True)
for i, (class_idx, count) in enumerate(zip(unique_train, counts_train)):
    print(f"{CLASS_LABELS[class_idx]}: {count} images")

print("\nTest Set Class Distribution:")
unique_test, counts_test = np.unique(y_test_raw, return_counts=True)
for i, (class_idx, count) in enumerate(zip(unique_test, counts_test)):
    print(f"{CLASS_LABELS[class_idx]}: {count} images")

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_cat, test_size=0.2, random_state=42, stratify=y_train_cat
)

print(f"\nFinal Dataset Split:")
print(f"Training set: {X_train_split.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

print("\nSample images from training set:")
visualize_sample_images(X_train, y_train_raw, CLASS_LABELS)

print("\n" + "=" * 60)
print("DATA AUGMENTATION SETUP")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train_split, y_train_split, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)
test_generator = test_datagen.flow(X_test, y_test_cat, batch_size=BATCH_SIZE)

print("Data augmentation setup complete!")

print("\n" + "=" * 60)
print("MODEL CREATION")
print("=" * 60)

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
model = create_cnn_model(input_shape, NUM_CLASSES)

print("Model Architecture:")
model.summary()

print("\n" + "=" * 60)
print("MODEL COMPILATION")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=8,
    min_lr=1e-7,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_facial_expression_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

print("Model compiled with callbacks setup!")

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

print("Starting model training...")
print(f"Training for maximum {EPOCHS} epochs")
print("Training will stop early if validation accuracy doesn't improve")

steps_per_epoch = len(X_train_split) // BATCH_SIZE
validation_steps = len(X_val) // BATCH_SIZE

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")

# Add visualization, evaluation, confusion matrix, saving, prediction, and summary code here similarly.
# (Omitted here for brevity but include from your original script.)

