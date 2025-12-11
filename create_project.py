"""
ML Project Generator
Creates standardized project structure for production-ready ML projects.
Usage: python create_project.py <project-name>
"""

import os
import sys
from pathlib import Path


def create_project_structure(project_name: str, base_path: str = "projects"):
    """
    Creates a standardized ML project structure.
    
    Args:
        project_name: Name of the project (will be slugified)
        base_path: Base directory where projects are stored
    """
    
    # Sanitize project name
    project_slug = project_name.lower().replace(" ", "-")
    project_path = Path(base_path) / project_slug
    
    if project_path.exists():
        print(f"❌ Project '{project_slug}' already exists!")
        return
    
    # Define structure
    structure = {
        "data": ["raw", "processed"],
        "notebooks": [],
        "src": [],
        "models": [],
        "outputs": ["figures", "reports"],
    }
    
    print(f"🚀 Creating project: {project_slug}")
    
    # Create directories
    for folder, subfolders in structure.items():
        folder_path = project_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {folder}/")
        
        for subfolder in subfolders:
            subfolder_path = folder_path / subfolder
            subfolder_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ Created {folder}/{subfolder}/")
    
    # Create .gitignore
    gitignore_content = """# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*.pkl
models/*.h5
models/*.pt
models/*.joblib
!models/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
.env

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDEs
.vscode/
.idea/

# ML Artifacts
mlruns/
wandb/
.neptune/

# OS
.DS_Store
Thumbs.db
"""
    
    (project_path / ".gitignore").write_text(gitignore_content)
    print(f"   ✓ Created .gitignore")
    
    # Create .gitkeep files
    (project_path / "data" / "raw" / ".gitkeep").touch()
    (project_path / "data" / "processed" / ".gitkeep").touch()
    (project_path / "models" / ".gitkeep").touch()
    
    # Create requirements.txt
    requirements_content = """# Core Data Science
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Visualization (Plotly First!)
plotly>=5.17.0
kaleido>=0.2.1  # For static image export

# Model Explainability
shap>=0.42.0

# Deployment
streamlit>=1.28.0

# Experiment Tracking (Optional - uncomment if needed)
# mlflow>=2.8.0
# wandb>=0.15.0

# Deep Learning (Optional - uncomment if needed)
# torch>=2.0.0
# tensorflow>=2.13.0

# Utilities
python-dotenv>=1.0.0
joblib>=1.3.0
"""
    
    (project_path / "requirements.txt").write_text(requirements_content)
    print(f"   ✓ Created requirements.txt")
    
    # Create starter notebook
    notebook_content = """# %% [markdown]
# # {PROJECT_NAME} - Exploratory Data Analysis
# 
# **Objective:** [Describe what you're trying to solve]
# 
# **Dataset:** [Describe the dataset]
# 
# ---

# %% [markdown]
# ## 1. Import Libraries

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set Plotly as default renderer
import plotly.io as pio
pio.renderers.default = "browser"

# %%
# Path configuration
DATA_RAW = Path("../data/raw")
DATA_PROCESSED = Path("../data/processed")
OUTPUTS = Path("../outputs/figures")

# %% [markdown]
# ## 2. Load Data

# %%
# df = pd.read_csv(DATA_RAW / "your_dataset.csv")
# df.head()

# %% [markdown]
# ## 3. Initial Exploration
# 
# Key questions:
# - What's the shape of the data?
# - Any missing values?
# - What are the data types?
# - Class balance (for classification)?

# %%
# Data shape and info
# print(f"Shape: {df.shape}")
# df.info()

# %%
# Missing values check
# missing = df.isnull().sum()
# missing[missing > 0]

# %% [markdown]
# ## 4. Visualization & Insights
# 
# **Remember:** Every plot should tell a story!

# %%
# Example: Distribution of target variable
# fig = px.histogram(
#     df, 
#     x="target_column",
#     title="Distribution of Target Variable",
#     color="target_column",
#     labels={"target_column": "Target"}
# )
# fig.show()

# %% [markdown]
# ## 5. Key Insights
# 
# **Findings:**
# 1. [Insight 1]
# 2. [Insight 2]
# 3. [Insight 3]
# 
# **Next Steps:**
# - [ ] Feature engineering ideas
# - [ ] Models to try
# - [ ] Evaluation metrics to track

""".replace("{PROJECT_NAME}", project_name.title())
    
    (project_path / "notebooks" / "01_eda.ipynb").write_text(
        convert_percent_to_ipynb(notebook_content)
    )
    print(f"   ✓ Created notebooks/01_eda.ipynb")
    
    # Create src templates
    preprocessing_content = '''"""
Data preprocessing utilities.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_and_split(
    filepath: str, 
    target_col: str, 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load data and split into train/test sets.
    
    Args:
        filepath: Path to the CSV file
        target_col: Name of the target column
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(filepath)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input dataframe
        strategy: Strategy for imputation ('mean', 'median', 'mode', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    if strategy == "drop":
        return df.dropna()
    elif strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    elif strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
'''
    
    (project_path / "src" / "preprocessing.py").write_text(preprocessing_content)
    print(f"   ✓ Created src/preprocessing.py")
    
    train_content = '''"""
Model training utilities.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path


def create_baseline_pipeline(estimator):
    """
    Create a baseline sklearn pipeline.
    
    Args:
        estimator: Sklearn estimator (e.g., LogisticRegression())
        
    Returns:
        Configured pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])


def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    """
    Train pipeline and print evaluation metrics.
    
    Args:
        pipeline: Sklearn pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Trained pipeline
    """
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    
    print("✅ Training complete!")
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"\\nTrain Accuracy: {train_score:.4f}")
    print(f"Test Accuracy:  {test_score:.4f}")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    print("\\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    return pipeline


def save_model(pipeline, filepath: str = "models/model.pkl"):
    """
    Save trained model to disk.
    
    Args:
        pipeline: Trained pipeline
        filepath: Output path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, filepath)
    print(f"💾 Model saved to {filepath}")
'''
    
    (project_path / "src" / "train.py").write_text(train_content)
    print(f"   ✓ Created src/train.py")
    
    visualization_content = '''"""
Visualization utilities using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Create an interactive confusion matrix using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (optional)
        
    Returns:
        Plotly figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f"Class {i}" for i in range(len(cm))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=600
    )
    
    return fig


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance from a trained model.
    
    Args:
        model: Trained sklearn model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        Plotly figure
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model doesn't have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(top_n)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'Importance': 'Feature Importance', 'Feature': ''},
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    return fig
'''
    
    (project_path / "src" / "visualization.py").write_text(visualization_content)
    print(f"   ✓ Created src/visualization.py")
    
    # Create Streamlit app template
    app_content = '''"""
Streamlit Deployment App
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="{PROJECT_NAME}",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 {PROJECT_NAME}")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Load model (cached)
@st.cache_resource
def load_model():
    model_path = Path("models/model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    else:
        return None

model = load_model()

if model is None:
    st.error("❌ Model not found! Train a model first.")
    st.stop()

# Main content
st.header("Make Predictions")

# TODO: Add input fields based on your features
# Example:
# col1, col2 = st.columns(2)
# with col1:
#     feature1 = st.number_input("Feature 1", value=0.0)
# with col2:
#     feature2 = st.number_input("Feature 2", value=0.0)

# if st.button("Predict", type="primary"):
#     # Create input dataframe
#     input_data = pd.DataFrame({
#         'feature1': [feature1],
#         'feature2': [feature2]
#     })
#     
#     # Make prediction
#     prediction = model.predict(input_data)[0]
#     proba = model.predict_proba(input_data)[0]
#     
#     # Display results
#     st.success(f"Prediction: {prediction}")
#     st.write(f"Confidence: {max(proba):.2%}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Vihaan Kulkarni")
'''.replace("{PROJECT_NAME}", project_name.title())
    
    (project_path / "app.py").write_text(app_content)
    print(f"   ✓ Created app.py")
    
    # Create README template
    readme_content = f"""# {project_name.title()}

> **Status:** 🚧 In Progress | ✅ Complete

## 🎯 Problem Statement

[Describe the problem you're solving. Why does this matter?]

## 📊 Dataset

- **Source:** [Link or description]
- **Size:** [Rows x Columns]
- **Target Variable:** [What are you predicting?]

## 🔍 Key Insights

### Insight 1: [Title]
[Screenshot of Plotly visualization]

**Finding:** [Describe what the data reveals]

### Insight 2: [Title]
[Screenshot of Plotly visualization]

**Finding:** [Describe what the data reveals]

## 🤖 Modeling Approach

### Baseline Model
- **Algorithm:** [e.g., Logistic Regression]
- **Accuracy:** XX%

### Final Model
- **Algorithm:** [e.g., Random Forest]
- **Accuracy:** XX%
- **Key Metrics:**
  - Precision: XX%
  - Recall: XX%
  - F1-Score: XX%

### Model Explainability
[SHAP summary plot or feature importance chart]

**Interpretation:** [Which features drove predictions?]

## 🚀 Deployment

Live app: [Streamlit link if deployed]

Run locally:
```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **Data:** Pandas, NumPy
- **Visualization:** Plotly
- **Modeling:** Scikit-Learn
- **Explainability:** SHAP
- **Deployment:** Streamlit

## 📁 Project Structure

```
{project_slug}/
├── data/              # Raw and processed data
├── notebooks/         # EDA and experiments
├── src/               # Production code
├── models/            # Saved models
├── outputs/           # Figures and reports
├── app.py             # Streamlit app
└── requirements.txt
```

## 🧪 Reproducibility

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebooks in order:
   - `01_eda.ipynb` - Exploratory Data Analysis
   - `02_modeling.ipynb` - Model Training

## 💡 Lessons Learned

- [Lesson 1]
- [Lesson 2]
- [Lesson 3]

## 🔮 Future Improvements

- [ ] [Improvement 1]
- [ ] [Improvement 2]
- [ ] [Improvement 3]

---

**Author:** Vihaan Kulkarni  
**Date:** {get_current_date()}
"""
    
    (project_path / "README.md").write_text(readme_content)
    print(f"   ✓ Created README.md")
    
    # Success message
    print(f"\n✨ Project '{project_slug}' created successfully!")
    print(f"\n📂 Location: {project_path.absolute()}")
    print(f"\n🚀 Next steps:")
    print(f"   1. cd {project_path}")
    print(f"   2. python -m venv .venv")
    print(f"   3. .venv\\Scripts\\activate  (Windows)")
    print(f"   4. pip install -r requirements.txt")
    print(f"   5. jupyter notebook notebooks/01_eda.ipynb")


def convert_percent_to_ipynb(content: str) -> str:
    """Convert percent format to Jupyter notebook JSON."""
    import json
    
    cells = []
    current_cell = {"lines": [], "type": "code"}
    
    for line in content.split("\n"):
        if line.startswith("# %% [markdown]"):
            if current_cell["lines"]:
                cells.append(current_cell)
            current_cell = {"lines": [], "type": "markdown"}
        elif line.startswith("# %%"):
            if current_cell["lines"]:
                cells.append(current_cell)
            current_cell = {"lines": [], "type": "code"}
        else:
            if current_cell["type"] == "markdown" and line.startswith("# "):
                current_cell["lines"].append(line[2:])
            else:
                current_cell["lines"].append(line)
    
    if current_cell["lines"]:
        cells.append(current_cell)
    
    notebook_cells = []
    for cell in cells:
        source = [line + "\n" for line in cell["lines"]]
        if source and not source[-1].endswith("\n"):
            source[-1] += "\n"
        
        notebook_cells.append({
            "cell_type": cell["type"],
            "metadata": {},
            "source": source,
            "outputs": [] if cell["type"] == "code" else None,
            "execution_count": None if cell["type"] == "code" else None
        })
        
        if notebook_cells[-1]["outputs"] is None:
            del notebook_cells[-1]["outputs"]
        if notebook_cells[-1]["execution_count"] is None:
            del notebook_cells[-1]["execution_count"]
    
    notebook = {
        "cells": notebook_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    return json.dumps(notebook, indent=2)


def get_current_date():
    """Get current date in YYYY-MM-DD format."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_project.py <project-name>")
        sys.exit(1)
    
    project_name = " ".join(sys.argv[1:])
    create_project_structure(project_name)
