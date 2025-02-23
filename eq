import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, spearmanr

# ✅ **Load Data**
df = pd.read_csv("equifax_insight_data.csv")

# ✅ **Rename Columns for Consistency**
df = df.rename(columns={
    "QTR_BOOKED": "Quarter_Booked",
    "epm_fico_score": "FICO",
    "epm_vantage_score_v_4_0": "Vantage4",
    "CIDS_score": "CIDS",
    "dq90plusever12": "Default"
})

df["Default"] = df["Default"].astype(int)
df.dropna(subset=["FICO", "Vantage4", "CIDS", "Default", "Quarter_Booked"], inplace=True)

# ✅ **Step 1: Compute Orthogonal Lift (R² & AUC Tests)**
results = []

for score_name in ["Vantage4", "CIDS"]:
    # **Linear Regression to Compute Residuals**
    X = df[["FICO"]]
    y = df[score_name]
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    
    r_squared = lin_reg.score(X, y)
    df[f"{score_name}_Residuals"] = y - lin_reg.predict(X)

    # **Logistic Regression Models for Default Prediction**
    X_full = df[["FICO", score_name]]
    X_base = df[["FICO"]]
    X_residual = df[[f"{score_name}_Residuals"]]
    y = df["Default"]

    # Train-Test Split
    X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.3, random_state=42)
    X_train_base, X_test_base, _, _ = train_test_split(X_base, y, test_size=0.3, random_state=42)
    X_train_residual, X_test_residual, _, _ = train_test_split(X_residual, y, test_size=0.3, random_state=42)

    # Fit Logistic Regression
    log_reg_full = LogisticRegression(max_iter=1000).fit(X_train_full, y_train)
    log_reg_base = LogisticRegression(max_iter=1000).fit(X_train_base, y_train)

    # Compute AUC Scores
    auc_full = roc_auc_score(y_test, log_reg_full.predict_proba(X_test_full)[:, 1])
    auc_base = roc_auc_score(y_test, log_reg_base.predict_proba(X_test_base)[:, 1])

    results.append({
        "Score Tested": score_name,
        "R²": round(r_squared, 3),
        "AUC-Full Model": round(auc_full, 3),
        "AUC-Base Model": round(auc_base, 3)
    })

# ✅ **Step 2: ROC Curve Comparison**
plt.figure(figsize=(10, 5))
fpr_full, tpr_full, _ = roc_curve(y_test, log_reg_full.predict_proba(X_test_full)[:, 1])
fpr_base, tpr_base, _ = roc_curve(y_test, log_reg_base.predict_proba(X_test_base)[:, 1])

plt.plot(fpr_full, tpr_full, label="Full Model (FICO + New Score)")
plt.plot(fpr_base, tpr_base, label="Base Model (FICO Only)")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

# ✅ **Step 3: KS Test for Risk Segmentation**
ks_score = ks_2samp(df[df["Default"] == 0]["CIDS"], df[df["Default"] == 1]["CIDS"])[0]
print(f"✅ KS Score (CIDS): {ks_score:.3f}")

# ✅ **Step 4: Default Rate by Score Buckets**
fico_bins = [300, 600, 650, 700, 750, 800, 850]
df["FICO_Bucket"] = pd.cut(df["FICO"], bins=fico_bins, labels=fico_bins[:-1])

default_rates = df.groupby("FICO_Bucket")["Default"].mean()

plt.figure(figsize=(10, 5))
sns.barplot(x=default_rates.index.astype(str), y=default_rates.values, color="red", alpha=0.7)
plt.xlabel("FICO Score Buckets")
plt.ylabel("Default Rate")
plt.title("Default Rate by FICO Score Buckets")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.show()

# ✅ **Step 5: Vintage-Based Default Rate Trends**
df["Quarter_Booked"] = pd.to_datetime(df["Quarter_Booked"])

vintage_defaults = df.groupby(df["Quarter_Booked"].dt.to_period("Q"))["Default"].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.lineplot(x="Quarter_Booked", y="Default", data=vintage_defaults, marker="o")
plt.xticks(rotation=45)
plt.xlabel("Quarter Booked")
plt.ylabel("Default Rate")
plt.title("Default Rate Trends by Quarter")
plt.grid()
plt.show()

# ✅ **Step 6: PSI for Score Buckets**
def compute_psi(base, current, bins=10):
    base_dist = base.value_counts(normalize=True, bins=bins).sort_index()
    current_dist = current.value_counts(normalize=True, bins=bins).sort_index()

    psi_values = (current_dist - base_dist) * np.log(current_dist / base_dist)
    return psi_values.sum()

base_period = df[df["Quarter_Booked"] < "2023-06"]["CIDS"]
current_period = df[df["Quarter_Booked"] >= "2023-06"]["CIDS"]

psi_cids = compute_psi(base_period, current_period)
print(f"✅ PSI for CIDS Score Buckets: {psi_cids:.3f}")

# ✅ **Step 7: Monotonicity Over Different Months**
monotonicity_results = []
for month, monthly_df in df.groupby(df["Quarter_Booked"].dt.to_period("M")):
    spearman_corr, _ = spearmanr(monthly_df["CIDS"], monthly_df["Default"])
    monotonicity_results.append({"Month": str(month), "Spearman Correlation": spearman_corr})

monotonicity_df = pd.DataFrame(monotonicity_results)

plt.figure(figsize=(10, 5))
sns.lineplot(x="Month", y="Spearman Correlation", data=monotonicity_df, marker="o")
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Spearman Correlation")
plt.title("Monotonicity Over Time")
plt.grid()
plt.show()

# ✅ **Store All Results**
import ace_tools as tools  
tools.display_dataframe_to_user(name="Equifax Insight Model Analysis", dataframe=results_df)
