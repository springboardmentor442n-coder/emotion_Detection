import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]   # moodmate/
DATA_DIR = BASE_DIR / "data"
IMG_DIR = DATA_DIR / "fer_images"
MODEL_DIR = BASE_DIR / "src" / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = 10
NUM_CLASSES = 7  # FER-2013 has 7 emotions

def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model

def main():
    # Data generators
    train_gen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    val_gen = ImageDataGenerator()

    train_ds = train_gen.flow_from_directory(
        IMG_DIR / "train", target_size=IMG_SIZE, batch_size=BATCH, class_mode="sparse"
    )
    val_ds = val_gen.flow_from_directory(
        IMG_DIR / "val", target_size=IMG_SIZE, batch_size=BATCH, class_mode="sparse"
    )

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cb = [
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "emotion_cnn_best.h5"),
            save_best_only=True,
            monitor="val_accuracy"
        ),
        callbacks.EarlyStopping(
            patience=3,
            monitor="val_accuracy",
            restore_best_weights=True
        )
    ]

    print("Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)

    # Fine-tune last layers
    print("Fine-tuning model...")
    for layer in model.layers[-30:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=4, callbacks=cb)

    # Save final model
    final_path = MODEL_DIR / "emotion_cnn_final.h5"
    model.save(final_path)
    print("Model saved:", final_path)

    # Evaluate on test set
    test_gen = ImageDataGenerator().flow_from_directory(
        IMG_DIR / "test", target_size=IMG_SIZE, batch_size=64,
        class_mode="sparse", shuffle=False
    )

    preds = model.predict(test_gen)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = test_gen.classes

    print(classification_report(true_labels, pred_labels))
    print("Confusion matrix:\n", confusion_matrix(true_labels, pred_labels))

    # Save predictions
    df = pd.DataFrame(preds, columns=[f"prob_{i}" for i in range(7)])
    df["pred_label"] = pred_labels
    df["true_label"] = true_labels
    df["filename"] = test_gen.filenames

    out_csv = DATA_DIR / "cnn_predictions.csv"
    df.to_csv(out_csv, index=False)
    print("Prediction CSV saved:", out_csv)

if __name__ == "__main__":
    main()
