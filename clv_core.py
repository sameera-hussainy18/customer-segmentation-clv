"""Prediction logic for the web app. There is no Streamlit code here, so it can be tested on its own.

Everything comes from the project notebook. data/ is written by make_app_data.py, which runs the notebook
and exports what it fitted:
  * Clustering uses the notebook's fitted parameters directly (centroids, medoids, GMM components,
    Mean Shift centres). Nothing is refitted, so every customer keeps exactly the notebook's cluster.
  * The CLV models are retrained here with the notebook's tuned hyperparameters, on the notebook's own
    training customers in the same order. This reproduces the notebook's predictions exactly
    (checked in tests/test_core.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent / "data"
RFM = ["Recency", "Frequency", "Monetary"]
MODEL_ORDER = ["K-Means", "K-Means++", "K-Medoids", "GMM", "Mean Shift"]
CLV_MODELS = [("Random Forest", "Untuned"), ("Random Forest", "Tuned"),
              ("Gradient Boosting", "Untuned"), ("Gradient Boosting", "Tuned")]


def load_meta(data_dir: Path = DATA_DIR) -> dict:
    return json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))


def load_table(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # round_trip: read every decimal back exactly as the notebook wrote it (deep trees are sensitive to 1-bit changes)
    return pd.read_csv(data_dir / name, float_precision="round_trip")


def as_matrix(rfm) -> np.ndarray:
    """Accept a DataFrame with Recency/Frequency/Monetary columns, one (R, F, M) triple or an (n, 3) array."""
    if isinstance(rfm, pd.DataFrame):
        return rfm[RFM].to_numpy(dtype=float)
    return np.atleast_2d(np.asarray(rfm, dtype=float))


def nearest_center(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Index of the closest centre (Euclidean) for every row: the K-Means / K-Medoids / Mean Shift rule."""
    return (((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)).argmin(axis=1)


def gmm_log_prob(X: np.ndarray, weights, means, precisions_cholesky) -> np.ndarray:
    """Weighted log-likelihood of every component: the formula GaussianMixture uses (covariance='full')."""
    n_features = X.shape[1]
    out = np.empty((X.shape[0], len(weights)))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_cholesky)):
        y = X @ prec_chol - mu @ prec_chol
        log_det = np.log(np.diag(prec_chol)).sum()
        out[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + (y ** 2).sum(axis=1)) + log_det + np.log(weights[k])
    return out


# ------------------------------------------------------------------------------------------------- clustering
class Segmenter:
    """Assigns customers to clusters with the parameters the notebook fitted (nothing is refitted here)."""

    def __init__(self, meta: dict):
        self.scale = np.asarray(meta["scaler"]["scale"], dtype=float)
        self.min = np.asarray(meta["scaler"]["min"], dtype=float)
        self.models = meta["models"]

    def to_scaled(self, rfm) -> np.ndarray:
        return as_matrix(rfm) * self.scale + self.min            # same arithmetic as MinMaxScaler.transform

    def to_original(self, scaled) -> np.ndarray:
        return (np.asarray(scaled, dtype=float) - self.min) / self.scale

    def _gmm(self, X: np.ndarray) -> np.ndarray:
        m = self.models["GMM"]
        return gmm_log_prob(X, np.asarray(m["weights"]), np.asarray(m["means_scaled"]),
                            np.asarray(m["precisions_cholesky"]))

    def predict(self, model_name: str, rfm) -> np.ndarray:
        X = self.to_scaled(rfm)
        model = self.models[model_name]
        if model["type"] == "gmm":
            return self._gmm(X).argmax(axis=1)
        return nearest_center(X, np.asarray(model["centers_scaled"], dtype=float))

    def predict_all(self, rfm) -> pd.DataFrame:
        return pd.DataFrame({name: self.predict(name, rfm) for name in MODEL_ORDER})

    def gmm_probabilities(self, rfm) -> np.ndarray:
        """GMM is a soft method: the probability of belonging to each component."""
        log_prob = self._gmm(self.to_scaled(rfm))
        log_prob -= log_prob.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        return prob / prob.sum(axis=1, keepdims=True)

    def centers(self, model_name: str) -> pd.DataFrame:
        """Cluster centres in original units (GMM: component means; K-Medoids: the medoid customers)."""
        model = self.models[model_name]
        key = "means_scaled" if model["type"] == "gmm" else "centers_scaled"
        table = pd.DataFrame(self.to_original(model[key]), columns=RFM)
        table.insert(0, "Cluster", range(len(table)))
        if "medoid_customer_ids" in model:
            table.insert(1, "Medoid customer", model["medoid_customer_ids"])
        return table


# ----------------------------------------------------------------------------- cluster profiles and labels
def profile_clusters(rfm_df: pd.DataFrame, labels) -> pd.DataFrame:
    """Customer count, share, mean R/F/M and revenue for every cluster (same as the notebook)."""
    data = rfm_df[RFM].copy()
    data["Cluster"] = np.asarray(labels)
    profile = (data.groupby("Cluster")
               .agg(**{"Customers": ("Recency", "size"), "Mean Recency": ("Recency", "mean"),
                       "Mean Frequency": ("Frequency", "mean"), "Mean Monetary": ("Monetary", "mean"),
                       "Total Revenue": ("Monetary", "sum")})
               .reset_index())
    profile.insert(2, "% of Customers", 100 * profile["Customers"] / profile["Customers"].sum())
    profile["% of Revenue"] = 100 * profile["Total Revenue"] / profile["Total Revenue"].sum()
    return profile


RECENCY_TEXT = {"Low": "purchased recently", "Medium": "purchased moderately recently",
                "High": "has not purchased for a long time"}
FREQUENCY_TEXT = {"Low": "orders rarely", "Medium": "orders occasionally", "High": "orders frequently"}
MONETARY_TEXT = {"Low": "spends little", "Medium": "spends a moderate amount", "High": "spends a lot"}
ACTIONS = {
    "High-value loyal": "Retain and reward: loyalty programme, early access to new products",
    "Valuable": "Upsell: bundles and personalised recommendations",
    "Active, average value": "Grow: cross-sell and volume discounts",
    "Recent, low spend": "Encourage repeat purchases: welcome series, second-order offer",
    "Recent, moderate spend": "Build loyalty: reorder reminders and member perks",
    "Occasional": "Nurture: regular newsletters and seasonal promotions",
    "At risk (was valuable)": "Re-engage quickly: personal offer before they lapse",
    "Lapsed": "Win back: reminder e-mail with a time-limited offer",
}


def level(value: float, overall_mean: float, low: float = 0.75, high: float = 1.25) -> str:
    """Low / Medium / High compared with the overall mean (the notebook's rule)."""
    ratio = value / overall_mean
    return "Low" if ratio < low else ("High" if ratio > high else "Medium")


def segment_name(r: str, f: str, m: str) -> str:
    """Short business name for a cluster from its R/F/M levels (a rule of thumb, used for display only)."""
    if r == "High":
        return "Lapsed" if f == "Low" and m == "Low" else "At risk (was valuable)"
    if f == "High" and m == "High":
        return "High-value loyal"
    if f == "High" or m == "High":
        return "Valuable"
    if f == "Medium" and m == "Medium":
        return "Active, average value"
    if r == "Low":
        return "Recent, low spend" if m == "Low" else "Recent, moderate spend"
    return "Occasional"


def describe_clusters(profile: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Plain-language description of every cluster, relative to the overall means (as in the notebook)."""
    overall = rfm_df[RFM].mean()
    rows = []
    for _, row in profile.iterrows():
        r = level(row["Mean Recency"], overall["Recency"])
        f = level(row["Mean Frequency"], overall["Frequency"])
        m = level(row["Mean Monetary"], overall["Monetary"])
        if r == "Low" and f == "High" and m == "High":
            text = "recently active, frequent and high-spending purchasing behaviour"
        elif r == "High" and f == "Low" and m == "Low":
            text = "less recent and lower-engagement purchasing behaviour"
        else:
            text = f"{RECENCY_TEXT[r]}, {FREQUENCY_TEXT[f]} and {MONETARY_TEXT[m]}"
        name = segment_name(r, f, m)
        rows.append({"Cluster": int(row["Cluster"]), "Segment": name, "Recency level": r, "Frequency level": f,
                     "Monetary level": m, "Description": text, "Suggested action": ACTIONS[name]})
    return pd.DataFrame(rows)


def clean_rfm_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Validate an uploaded table. Needs Recency, Frequency, Monetary columns (any capitalisation); an optional
    CustomerID is kept. Rows with missing, non-numeric or impossible values are dropped and counted.
    Raises ValueError with a readable message when nothing usable is left."""
    columns = {str(c).strip().lower(): c for c in frame.columns}
    missing = [f for f in RFM if f.lower() not in columns]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Expected Recency, Frequency, Monetary.")
    out = pd.DataFrame(index=frame.index)
    if "customerid" in columns:
        out["CustomerID"] = frame[columns["customerid"]].astype(str)
    for feature in RFM:
        out[feature] = pd.to_numeric(frame[columns[feature.lower()]], errors="coerce")
    finite = np.isfinite(out[RFM].to_numpy(dtype=float)).all(axis=1)
    valid = finite & (out["Recency"] >= 0) & (out["Frequency"] >= 1) & (out["Monetary"] >= 0)
    out = out[valid].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid rows: Recency ≥ 0, Frequency ≥ 1 and Monetary ≥ 0 are required.")
    return out, int((~valid).sum())


def out_of_range(values: dict, bounds: dict) -> list[str]:
    """Messages for inputs outside the given {feature: (low, high)} bounds."""
    messages = []
    for feature, (low, high) in bounds.items():
        value = values[feature]
        if value < low or value > high:
            messages.append(f"{feature} = {value:,.2f} is outside {low:,.2f} … {high:,.2f}")
    return messages


# ------------------------------------------------------------------------------------------------ CLV models
class CLVPredictor:
    """Random Forest and Gradient Boosting CLV models, retrained exactly as in the notebook.

    Inputs: Recency, Frequency and Monetary measured BEFORE the cutoff, plus the customer's K-Means++
    segment (one-hot). Target: net revenue in the 100 days after the cutoff.
    """

    def __init__(self, meta: dict, train: pd.DataFrame, test: pd.DataFrame):
        clv = meta["clv"]
        self.settings = clv
        self.random_state = meta["random_state"]
        self.k = int(clv["k"])
        self.feature_columns = clv["feature_columns"]
        self.scale = np.asarray(clv["scaler"]["scale"], dtype=float)
        self.min = np.asarray(clv["scaler"]["min"], dtype=float)
        self.centers = np.asarray(clv["segment_centers_scaled"], dtype=float)
        self.train, self.test = train, test
        self.models: dict[tuple[str, str], object] = {}

    # ---- segment + features ----
    def segment(self, rfm) -> np.ndarray:
        """K-Means++ segment used as a model input (the segment model was fitted on training customers only)."""
        return nearest_center(as_matrix(rfm) * self.scale + self.min, self.centers)

    def features(self, rfm, segments=None) -> pd.DataFrame:
        X = pd.DataFrame(as_matrix(rfm), columns=RFM)
        segments = self.segment(rfm) if segments is None else np.asarray(segments)
        for c in range(self.k):
            X[f"Cluster_{c}"] = (segments == c).astype(int)
        return X[self.feature_columns]

    # ---- training ----
    def make_model(self, name: str, variant: str):
        params = dict(self.settings["tuned_params" if variant == "Tuned" else "untuned_params"][name])
        if name == "Random Forest":
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1, **params)
        return GradientBoostingRegressor(random_state=self.random_state, **params)

    def fit(self) -> "CLVPredictor":
        X = self.features(self.train, self.train["Cluster"])
        y = self.train["CLV_Target"]
        for name, variant in CLV_MODELS:
            self.models[(name, variant)] = self.make_model(name, variant).fit(X, y)
        return self

    # ---- prediction ----
    def predict(self, rfm) -> pd.DataFrame:
        """Segment + the raw output of all four models (the target is never negative; clip for display)."""
        segments = self.segment(rfm)
        X = self.features(rfm, segments)
        out = pd.DataFrame({"Segment": segments})
        for (name, variant), model in self.models.items():
            out[f"{name} ({variant})"] = model.predict(X)
        return out

    def test_predictions(self) -> pd.DataFrame:
        X = self.features(self.test, self.test["Cluster"])
        out = self.test[["CustomerID"] + RFM + ["Cluster", "CLV_Target"]].copy()
        for (name, variant), model in self.models.items():
            out[f"{name} ({variant})"] = model.predict(X)
        return out

    def evaluate(self) -> pd.DataFrame:
        """MAE, MSE, RMSE and R² on the test customers (never seen during training or tuning)."""
        predictions = self.test_predictions()
        y = predictions["CLV_Target"]
        rows = []
        for name, variant in CLV_MODELS:
            p = predictions[f"{name} ({variant})"]
            mse = mean_squared_error(y, p)
            rows.append({"Model": name, "Tuned/Untuned": variant, "MAE": mean_absolute_error(y, p), "MSE": mse,
                         "RMSE": float(np.sqrt(mse)), "R²": r2_score(y, p)})
        return pd.DataFrame(rows)

    def feature_importance(self, variant: str = "Tuned") -> pd.DataFrame:
        rows = []
        for name in ["Random Forest", "Gradient Boosting"]:
            model = self.models[(name, variant)]
            rows += [{"Model": name, "Feature": f, "Importance": v}
                     for f, v in zip(self.feature_columns, model.feature_importances_)]
        return pd.DataFrame(rows)

    def training_bounds(self) -> dict:
        """Range of each input among the training customers: outside it, the model extrapolates."""
        return {f: (float(self.train[f].min()), float(self.train[f].max())) for f in RFM}

    def same_length_history(self, monetary: float) -> float:
        """Pre-cutoff revenue rescaled to the length of the prediction window (the notebook's 'historical CLV')."""
        return monetary * self.settings["prediction_days"] / self.settings["observation_days"]
