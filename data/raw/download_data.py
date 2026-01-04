
import pandas as pd
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]

os.makedirs("data/raw", exist_ok=True)
df = pd.read_csv(URL, header=None, names=COLUMNS)
df.replace("?", pd.NA, inplace=True)
df.to_csv("data/raw/heart.csv", index=False)
print("Dataset downloaded successfully")
