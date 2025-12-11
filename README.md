# 🤖 Vihaan's ML Engineering Workspace

> **Mission:** Build production-ready ML projects with world-class data storytelling.

This workspace is designed to accelerate ML project development with standardized templates, reusable utilities, and a focus on **explainability** and **deployment**.

---

## 📁 Workspace Structure

```
ML/
├── projects/               # Individual ML projects live here
│   ├── project-1/
│   ├── project-2/
│   └── ...
├── templates/              # Reusable code templates
├── utils/                  # Shared utilities across all projects
│   ├── visualization_helpers.py    # Plotly visualization templates
│   └── model_evaluation.py         # Model evaluation utilities
├── create_project.py       # 🚀 Project generator script
└── README.md              # You are here
```

---

## 🚀 Quick Start: Create a New Project

Run the project generator to scaffold a production-ready ML project:

```powershell
python create_project.py "Project Name"
```

### What Gets Created:

```
project-name/
├── data/
│   ├── raw/                # Store raw datasets here (gitignored)
│   └── processed/          # Cleaned/transformed data
├── notebooks/
│   └── 01_eda.ipynb        # Starter EDA notebook with Plotly templates
├── src/
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── train.py            # Model training utilities
│   └── visualization.py    # Project-specific visualizations
├── models/                 # Saved models (gitignored)
├── outputs/
│   ├── figures/            # Saved Plotly charts
│   └── reports/            # Generated reports
├── app.py                  # Streamlit deployment template
├── requirements.txt        # Dependencies
├── .gitignore              # Proper ML gitignore
└── README.md               # Portfolio-ready documentation template
```

---

## 🎯 Philosophy

### 1. **Code for Production**
- No messy notebooks with 500 lines of code
- Refactor complex logic into `src/` modules
- Use sklearn Pipelines for reproducibility

### 2. **Visualization First**
- Default to **Plotly Express** (interactive > static)
- Every chart must tell a story
- Include titles, axis labels, and hover data

### 3. **Explainability is Mandatory**
- Always explain model predictions (SHAP, feature importance)
- Visualize confusion matrices and error analysis
- No black box models

### 4. **Deployment Mindset**
- Every project gets a `app.py` Streamlit template
- Models saved with `joblib` for easy loading
- README formatted for portfolio/resume

---

## 🛠️ Shared Utilities

### `utils/visualization_helpers.py`
Pre-built Plotly templates for common charts:
- `create_distribution_plot()` - Histograms with KDE overlays
- `create_correlation_heatmap()` - Interactive correlation matrices
- `create_scatter_with_trend()` - Scatter plots with trendlines
- `create_grouped_bar_chart()` - Grouped bar charts
- `create_time_series_plot()` - Time series with range slider

### `utils/model_evaluation.py`
Comprehensive evaluation functions:
- `evaluate_classifier()` - Confusion matrix, ROC curve, classification report
- `evaluate_regressor()` - R², RMSE, MAE, residual plots
- `print_evaluation_summary()` - Formatted metric summary

**Usage Example:**
```python
from utils.model_evaluation import evaluate_classifier

results = evaluate_classifier(y_test, y_pred, y_proba)
results['confusion_matrix_fig'].show()
print_evaluation_summary(results)
```

---

## 📋 Project Workflow

### Phase 1: Setup
```powershell
python create_project.py "Customer Churn Prediction"
cd projects/customer-churn-prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Phase 2: EDA
1. Add dataset to `data/raw/`
2. Open `notebooks/01_eda.ipynb`
3. Use Plotly for all visualizations
4. Check for:
   - Missing values
   - Class imbalance
   - Feature distributions
   - Correlations

### Phase 3: Modeling
1. Create baseline model first (e.g., DummyClassifier)
2. Refactor training code into `src/train.py`
3. Use sklearn Pipelines
4. Track experiments (consider MLflow)

### Phase 4: Evaluation
1. Generate confusion matrix
2. Plot SHAP values
3. Analyze misclassifications
4. Save best model to `models/`

### Phase 5: Deployment
1. Update `app.py` with input fields
2. Test locally: `streamlit run app.py`
3. Deploy to Streamlit Cloud (optional)

### Phase 6: Documentation
1. Fill out `README.md` with:
   - Problem statement
   - Key insights (with screenshots)
   - Model metrics
   - Deployment link
2. Use "Problem → Method → Insight → Impact" structure

---

## 📦 Standard Dependencies

Every project includes:
- **Data:** `pandas`, `numpy`
- **Visualization:** `plotly`, `kaleido`
- **Modeling:** `scikit-learn`
- **Explainability:** `shap`
- **Deployment:** `streamlit`

Optional (uncomment in `requirements.txt` if needed):
- **Experiment Tracking:** `mlflow`, `wandb`
- **Deep Learning:** `torch`, `tensorflow`

---

## 💡 Pro Tips

1. **Run baseline first:** Always compare against a simple model
2. **Plotly over Matplotlib:** Interactive charts reveal more insights
3. **Document as you go:** Fill README during the project, not after
4. **Save figures:** Use `fig.write_html()` to preserve interactivity
5. **Version control:** Commit after each major milestone

---

## 🎓 Learning Resources

- [Plotly Documentation](https://plotly.com/python/)
- [SHAP Tutorial](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)

---

## 📊 Portfolio Goals

- ✅ 1 high-quality project per week
- ✅ Every project deployed with Streamlit
- ✅ README formatted for resume/GitHub
- ✅ Interactive visualizations (no static PNGs)
- ✅ Model explainability included

---

**Built by Vihaan Kulkarni**  
*Senior ML Engineer & Data Storyteller*
