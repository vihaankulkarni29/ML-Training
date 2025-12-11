# 🚀 Future Visions for Model Evaluation Utilities

This document captures advanced ideas and best practices for evolving our ML evaluation scripts, inspired by industry standards and the GeeksforGeeks article on ML metrics.

---

## 1. Classification Metrics: Planned Additions

- **Logarithmic Loss (Log Loss):**
  - Measures uncertainty/confidence of probabilistic predictions.
  - Useful for multi-class and when model confidence matters.
- **Matthews Correlation Coefficient (MCC):**
  - Robust for imbalanced datasets.
- **Cohen’s Kappa:**
  - Measures agreement, correcting for chance.
- **Specificity (True Negative Rate), FPR, FNR:**
  - Especially for medical/fraud applications.
- **Precision-Recall Curve & Average Precision:**
  - Key for imbalanced datasets.
- **Customizable metric selection:**
  - Allow user to choose which metrics to compute.

## 2. Regression Metrics: Planned Additions

- **Root Mean Squared Logarithmic Error (RMSLE):**
  - For targets with large range or exponential growth.
- **Mean Absolute Percentage Error (MAPE):**
  - Error as a percentage, useful for business/finance.
- **Explained Variance Score:**
  - Proportion of variance explained by the model.

## 3. Clustering Metrics (For Future Unsupervised Tasks)

- **Silhouette Score**
- **Davies-Bouldin Index**

## 4. Visualization Enhancements

- **Precision-Recall Curve (Plotly)**
- **Residual Distribution Plot (Regression)**
- **Summary Table for All Metrics**

## 5. Best Practices

- Report all metrics in a single dictionary for easy logging.
- Visualize more than just confusion matrix and ROC.
- Add summary table for README/reporting.

---

**This file is a living roadmap. As the portfolio matures, revisit and implement these enhancements for even more robust, explainable, and professional ML projects.**
