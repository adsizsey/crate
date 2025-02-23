import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# ✅ **Compute ROC Curve for Raw Scores**
plt.figure(figsize=(10, 5))

for score_name in ["FICO", "Vantage4", "CIDS"]:
    fpr, tpr, _ = roc_curve(df["Default"], -df[score_name])  # Multiply by -1
    auc_value = roc_auc_score(df["Default"], -df[score_name])

    plt.plot(fpr, tpr, label=f"{score_name} (AUC = {auc_value:.3f})")

# ✅ **Plotting the ROC Curves**
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random Guess Line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Raw Scores, Adjusted for Direction)")
plt.legend()
plt.grid()
plt.show()
