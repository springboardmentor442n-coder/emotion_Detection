import pandas as pd
import numpy as np
import os
from PIL import Image

FER = "data/fer2013.csv"
OUT_DIR = "data/fer_images"

EMOTIONS = {
    0:"angry", 1:"disgust", 2:"fear",
    3:"happy", 4:"sad", 5:"surprise", 6:"neutral"
}

def make_dirs():
    for split in ["train", "val", "test"]:
        for e in EMOTIONS.values():
            os.makedirs(f"{OUT_DIR}/{split}/{e}", exist_ok=True)

def convert():
    if not os.path.exists(FER):
        print("❌ fer2013.csv not found in data/")
        return

    df = pd.read_csv(FER)
    make_dirs()

    for i, row in df.iterrows():
        usage = row["Usage"]
        
        # CORRECT mapping
        if usage == "Training":
            split = "train"
        elif usage == "PublicTest":
            split = "val"
        elif usage == "PrivateTest":
            split = "test"
        else:
            continue

        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels)
        label = EMOTIONS[row["emotion"]]

        img.save(f"{OUT_DIR}/{split}/{label}/{i}.png")

    print("✔ Conversion complete!")
    print("✔ Train images: ", sum(len(files) for _, _, files in os.walk(f"{OUT_DIR}/train")))
    print("✔ Val images:   ", sum(len(files) for _, _, files in os.walk(f"{OUT_DIR}/val")))
    print("✔ Test images:  ", sum(len(files) for _, _, files in os.walk(f"{OUT_DIR}/test")))

if __name__ == "__main__":
    convert()
