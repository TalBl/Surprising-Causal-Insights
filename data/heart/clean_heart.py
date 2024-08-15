import pandas as pd

def cast_age(value):
    if value < 30:
        return "20-30"
    if 30 <= value < 40:
        return "30-40"
    if 40 <= value < 50:
        return "40-50"
    if 50 <= value < 60:
        return "50-60"
    if 60 <= value < 70:
        return "30-40"
    if 70 <= value < 80:
        return "70-80"

def cast_chol(value):
    if value < 120:
        return "<120"
    if 120 <= value < 180:
        return "120-180"
    if 180 <= value < 240:
        return "180-240"
    if 240 <= value < 300:
        return "240-300"
    if 300 <= value < 360:
        return "300-360"
    return ">360"

df = pd.read_csv("heart.csv")
df["Age"] = df["Age"].apply(cast_age)
df["Cholesterol"] = df["Cholesterol"].apply(cast_chol)
one_hot_encoded = pd.get_dummies(df, columns=['ChestPainType'])
one_hot_encoded["ChestPainType_TA"] = one_hot_encoded["ChestPainType_TA"].astype(int)
one_hot_encoded["ChestPainType_ASY"] = one_hot_encoded["ChestPainType_ASY"].astype(int)
one_hot_encoded.to_csv("clean_data.csv", index=False)
