# 🧮 Tip Prediction — Streamlit App (Load-only)

This repository contains a **minimal Streamlit app** that loads a pre-trained `model.pkl` and predicts the restaurant **tip** based on user inputs. No training happens at app startup — it is fast and deployment-friendly.

## Project Structure
```
.
├─ app.py          # Streamlit app (loads model.pkl only)
├─ model.pkl       # Pre-trained RandomForestRegressor (create in notebook)
├─ requirements.txt
└─ README.md
```

## How to run locally
1) Create/activate your environment and install dependencies:
```bash
pip install -r requirements.txt
```
2) Make sure `model.pkl` is in the same folder as `app.py`.
3) Start the app:
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

## How to create `model.pkl` (Notebook snippet)
Train in your notebook and export the model:
```python
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_excel("tip.xlsx")
df["sex"]    = df["sex"].map({"Female": 0, "Male": 1})
df["smoker"] = df["smoker"].map({"No": 0, "Yes": 1})
df["day"]    = df["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
df["time"]   = df["time"].map({"Lunch": 0, "Dinner": 1})
df["avg_bill_per_person"] = df["total_bill"] / df["size"].replace(0, np.nan)
df["avg_bill_per_person"] = df["avg_bill_per_person"].fillna(df["total_bill"])
df["weekend"] = df["day"].isin([2, 3]).astype(int)

X = df[["total_bill","sex","smoker","day","time","size","avg_bill_per_person","weekend"]]
y = df["tip"]

model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("✅ Saved model.pkl")
```

> ⚠️ **Important**: The **feature order** in `app.py` must match the training order above.

## Deployment (GitHub / Hugging Face)
- **GitHub**: Commit `app.py`, `requirements.txt`, `model.pkl`, and `README.md` to your repo.
- **Hugging Face Spaces (Streamlit)**: Upload the same files. Spaces will detect Streamlit automatically and run the app.

## References

- Dataset: The classic **Tips dataset** (originally from Seaborn).
- Inspiration: Some ideas were inspired by [AmanXAI - Waiter Tips Prediction with ML](https://amanxai.com/2022/02/01/waiter-tips-prediction-with-machine-learning/).


## Footnote
**Made with ❤️ by Emr7y | Model: RandomForest | Data: tips (Seaborn) / tip.xlsx**
