# ============================================================
#  Thiranex Project 3 — Predictive Analytics Using Historical Data
#  Topic: House Price Prediction (Regression)
#  Author: Tejas
#  Description: Predicts house prices using Linear Regression
#               and Random Forest, with preprocessing,
#               evaluation metrics, and visualizations.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# STEP 1: LOAD OR GENERATE DATASET
# ─────────────────────────────────────────────
# If you have the Kaggle House Prices CSV, place it in the same
# folder as this script and name it "house_prices.csv".
# Otherwise, a synthetic dataset will be generated automatically.

DATASET_FILE = "house_prices.csv"

def load_or_generate_data():
    if os.path.exists(DATASET_FILE):
        print(f"[INFO] Loading dataset from '{DATASET_FILE}'...")
        df = pd.read_csv(DATASET_FILE)
        # Use a subset of useful Kaggle columns if available
        kaggle_cols = [
            "GrLivArea", "BedroomAbvGr", "FullBath", "GarageCars",
            "YearBuilt", "OverallQual", "TotalBsmtSF", "Neighborhood",
            "SalePrice"
        ]
        available = [c for c in kaggle_cols if c in df.columns]
        df = df[available].copy()
        df.rename(columns={"SalePrice": "Price"}, inplace=True)
        print(f"[INFO] Loaded {len(df)} rows with columns: {list(df.columns)}")
    else:
        print("[INFO] Dataset not found. Generating synthetic house price data...")
        np.random.seed(42)
        n = 1000
        neighborhoods = ["Downtown", "Suburbs", "Uptown", "Riverside", "Eastside"]
        df = pd.DataFrame({
            "GrLivArea":      np.random.randint(500, 4000, n),
            "BedroomAbvGr":   np.random.randint(1, 6, n),
            "FullBath":       np.random.randint(1, 4, n),
            "GarageCars":     np.random.randint(0, 4, n),
            "YearBuilt":      np.random.randint(1950, 2023, n),
            "OverallQual":    np.random.randint(1, 11, n),
            "TotalBsmtSF":    np.random.randint(0, 2500, n),
            "Neighborhood":   np.random.choice(neighborhoods, n),
        })
        # Simulate price with some noise
        df["Price"] = (
            df["GrLivArea"]    * 80 +
            df["OverallQual"]  * 10000 +
            df["GarageCars"]   * 8000 +
            df["FullBath"]     * 5000 +
            df["TotalBsmtSF"]  * 20 +
            (df["YearBuilt"] - 1950) * 300 +
            np.random.normal(0, 15000, n)
        ).clip(50000, 800000).round(-2)
        print(f"[INFO] Generated {n} synthetic records.")
    return df


# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    print("\n[STEP 2] Preprocessing data...")
    print(f"  Shape before: {df.shape}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded categorical column: '{col}'")

    # Impute missing numeric values with median
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Remove clear outliers (prices beyond 3 std deviations)
    z_scores = np.abs((df_imputed["Price"] - df_imputed["Price"].mean()) / df_imputed["Price"].std())
    before = len(df_imputed)
    df_imputed = df_imputed[z_scores < 3].reset_index(drop=True)
    print(f"  Outliers removed: {before - len(df_imputed)} rows")
    print(f"  Shape after: {df_imputed.shape}")

    return df_imputed


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING & SPLIT
# ─────────────────────────────────────────────

def prepare_features(df):
    print("\n[STEP 3] Preparing features...")

    # Feature: House Age
    if "YearBuilt" in df.columns:
        df["HouseAge"] = 2024 - df["YearBuilt"]

    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print(f"  Features used    : {list(X.columns)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler


# ─────────────────────────────────────────────
# STEP 4: TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    print("\n[STEP 4] Training models...")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("  ✔ Linear Regression trained.")

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("  ✔ Random Forest Regressor trained.")

    return lr, rf


# ─────────────────────────────────────────────
# STEP 5: EVALUATE MODELS
# ─────────────────────────────────────────────

def evaluate(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  [{name}]")
    print(f"    MAE  : ₱{mae:,.2f}")
    print(f"    RMSE : ₱{rmse:,.2f}")
    print(f"    R²   : {r2:.4f}")

    return y_pred, {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ─────────────────────────────────────────────
# STEP 6: VISUALIZATIONS
# ─────────────────────────────────────────────

def visualize(y_test, lr_pred, rf_pred, rf_model, feature_names, results):
    print("\n[STEP 6] Generating visualizations...")
    y_test = np.array(y_test)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("House Price Prediction — Thiranex Project 3", fontsize=16, fontweight="bold")

    # --- Plot 1: Actual vs Predicted (Linear Regression) ---
    ax1 = axes[0, 0]
    ax1.scatter(y_test, lr_pred, alpha=0.5, color="#4C72B0", edgecolors="none", s=20)
    lim = [min(y_test.min(), lr_pred.min()), max(y_test.max(), lr_pred.max())]
    ax1.plot(lim, lim, "r--", linewidth=1.5, label="Perfect Fit")
    ax1.set_xlabel("Actual Price")
    ax1.set_ylabel("Predicted Price")
    ax1.set_title("Linear Regression: Actual vs Predicted")
    ax1.legend()

    # --- Plot 2: Actual vs Predicted (Random Forest) ---
    ax2 = axes[0, 1]
    ax2.scatter(y_test, rf_pred, alpha=0.5, color="#55A868", edgecolors="none", s=20)
    ax2.plot(lim, lim, "r--", linewidth=1.5, label="Perfect Fit")
    ax2.set_xlabel("Actual Price")
    ax2.set_ylabel("Predicted Price")
    ax2.set_title("Random Forest: Actual vs Predicted")
    ax2.legend()

    # --- Plot 3: Residuals (Random Forest) ---
    ax3 = axes[1, 0]
    residuals = y_test - rf_pred
    ax3.hist(residuals, bins=40, color="#C44E52", edgecolor="white", alpha=0.8)
    ax3.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax3.set_xlabel("Residual (Actual − Predicted)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Random Forest: Residual Distribution")

    # --- Plot 4: Feature Importances (Random Forest) ---
    ax4 = axes[1, 1]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True).tail(8)
    ax4.barh(feat_df["Feature"], feat_df["Importance"], color="#8172B2")
    ax4.set_xlabel("Importance Score")
    ax4.set_title("Top Feature Importances (Random Forest)")

    plt.tight_layout()
    out_file = "house_price_prediction_results.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"  ✔ Saved visualization → '{out_file}'")
    plt.show()

    # --- Model Comparison Bar Chart ---
    fig2, ax = plt.subplots(figsize=(7, 4))
    labels  = [r["model"] for r in results]
    r2_vals = [r["R2"]    for r in results]
    bars = ax.bar(labels, r2_vals, color=["#4C72B0", "#55A868"], width=0.4)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² Score")
    ax.set_title("Model Comparison — R² Score")
    fig2.tight_layout()
    out_file2 = "model_comparison.png"
    plt.savefig(out_file2, dpi=150, bbox_inches="tight")
    print(f"  ✔ Saved model comparison → '{out_file2}'")
    plt.show()


# ─────────────────────────────────────────────
# STEP 7: SAMPLE PREDICTION
# ─────────────────────────────────────────────

def sample_prediction(rf_model, scaler, feature_names):
    print("\n[STEP 7] Sample Prediction")
    print("  Using a sample house:")

    # Build a sample with the same features
    sample = {col: 0 for col in feature_names}
    sample.update({
        "GrLivArea":    1800,
        "BedroomAbvGr": 3,
        "FullBath":     2,
        "GarageCars":   2,
        "YearBuilt":    2005,
        "OverallQual":  7,
        "TotalBsmtSF":  900,
        "Neighborhood": 1,   # encoded
        "HouseAge":     19,
    })

    sample_df = pd.DataFrame([{col: sample.get(col, 0) for col in feature_names}])
    sample_scaled = scaler.transform(sample_df)
    predicted_price = rf_model.predict(sample_scaled)[0]

    print(f"  Living Area   : {sample.get('GrLivArea', 'N/A')} sq ft")
    print(f"  Bedrooms      : {sample.get('BedroomAbvGr', 'N/A')}")
    print(f"  Bathrooms     : {sample.get('FullBath', 'N/A')}")
    print(f"  Garage Cars   : {sample.get('GarageCars', 'N/A')}")
    print(f"  Year Built    : {sample.get('YearBuilt', 'N/A')}")
    print(f"  Overall Qual  : {sample.get('OverallQual', 'N/A')}/10")
    print(f"\n  ➤ Predicted Price : ₱{predicted_price:,.2f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Thiranex Project 3 — House Price Prediction")
    print("=" * 55)

    # 1. Load data
    df = load_or_generate_data()

    # 2. Preprocess
    df = preprocess(df)

    # 3. Features & split
    X_train, X_test, y_train, y_test, feat_names, scaler = prepare_features(df)

    # 4. Train
    lr_model, rf_model = train_models(X_train, y_train)

    # 5. Evaluate
    print("\n[STEP 5] Evaluating models...")
    lr_pred, lr_stats = evaluate(lr_model, "Linear Regression", X_test, y_test)
    rf_pred, rf_stats = evaluate(rf_model, "Random Forest",     X_test, y_test)

    # 6. Visualize
    visualize(y_test, lr_pred, rf_pred, rf_model, feat_names, [lr_stats, rf_stats])

    # 7. Sample prediction
    sample_prediction(rf_model, scaler, feat_names)

    print("\n" + "=" * 55)
    print("  ✅ Project 3 Complete!")
    print("=" * 55)
