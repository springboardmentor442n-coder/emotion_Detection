# src/utils/fer_to_images.py
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import sys

def fer_csv_to_images(csv_path, out_dir, img_size=(224,224)):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    if not csv_path.exists():
        print("FER CSV not found at", csv_path)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    for i, row in df.iterrows():
        emotion = int(row['emotion'])
        pixels = np.fromstring(row['pixels'], sep=' ', dtype='uint8').reshape(48,48)
        img = Image.fromarray(pixels).convert('RGB').resize(img_size)

        usage = row.get('Usage', 'Training')
        subset = 'train' if 'Train' in usage else ('val' if 'PublicTest' in usage else 'test')

        save_dir = out_dir / subset / str(emotion)
        save_dir.mkdir(parents=True, exist_ok=True)
        img.save(save_dir / f"{i}.jpg")

    print("Conversion complete:", out_dir)

if __name__ == "__main__":
    fer_csv_to_images(sys.argv[1], sys.argv[2])
