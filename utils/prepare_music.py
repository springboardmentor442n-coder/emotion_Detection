# src/utils/prepare_music.py
import pandas as pd
from pathlib import Path
p = Path("data/music.csv")
df = pd.read_csv(p)
df['tags'] = df['tags'].fillna('').str.lower().str.replace('[^a-z0-9 ]','', regex=True)
# simple split and unique tag list as string
df['tags_clean'] = df['tags'].apply(lambda s: ' '.join(sorted(set(s.split()))))
out = Path("data/music_clean.csv")
df.to_csv(out, index=False)
print("Saved", out)
