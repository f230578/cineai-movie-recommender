import os
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ann_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")


def train_ann_model(df):
    X = df[["runtime", "year", "popularity"]].astype(float).values
    y = df["rating"].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=1000,
            random_state=42
        )
        model.fit(X_scaled, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    preds = model.predict(X_scaled)
    mae = round(float(np.mean(np.abs(preds - y))), 4)
    return model, scaler, mae


def load_ann_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None


def predict_ratings(df):
    model, scaler = load_ann_model()
    if model is None:
        model, scaler, _ = train_ann_model(df)

    X = df[["runtime", "year", "popularity"]].astype(float).values
    X_scaled = scaler.transform(X)
    preds = np.clip(model.predict(X_scaled), 1.0, 10.0)

    df = df.copy()
    df["ann_predicted_rating"] = np.round(preds, 2)
    return df


def is_model_trained():
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
