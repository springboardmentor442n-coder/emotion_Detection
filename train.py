import tensorflow as tf
from model import build_model

DATA = "data/fer_images"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATA}/train", image_size=(48,48), batch_size=64)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    f"{DATA}/val", image_size=(48,48), batch_size=64)

model = build_model()

model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("models/emotion_model.h5")

print("✔ Model trained & saved!")
