from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------------------------------------------------------------------
# Feature extraction (self-contained copy from src/features.py)
# ---------------------------------------------------------------------------

def get_properties(seq: Any) -> Dict[str, float]:
    """Compute peptide physicochemical properties safely."""
    if not isinstance(seq, str) or len(seq.strip()) == 0:
        return {
            "length": 0,
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "positive_charge": 0,
        }

    seq = seq.strip()
    try:
        analysis = ProteinAnalysis(seq)
        return {
            "length": len(seq),
            "mol_weight": float(analysis.molecular_weight()),
            "aromaticity": float(analysis.aromaticity()),
            "instability_index": float(analysis.instability_index()),
            "isoelectric_point": float(analysis.isoelectric_point()),
            "gravy": float(analysis.gravy()),
            "positive_charge": seq.count("K") + seq.count("R"),
        }
    except Exception:
        return {
            "length": 0,
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "positive_charge": 0,
        }


@st.cache_resource
def load_model():
    """Load trained MIC regressor from default or project path."""
    primary = Path("models/mic_predictor.pkl")
    fallback = Path("projects/MIC Regression/models/mic_predictor.pkl")
    model_path = primary if primary.exists() else fallback

    if not model_path.exists():
        st.error(
            "Model file not found. Checked:\n"
            f"- {primary}\n"
            f"- {fallback}"
        )
        return None

    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Error loading model: {exc}")
        return None


def interpret_mic(mic_uM: float) -> str:
    if mic_uM < 2:
        return "💎 Excellent Candidate (High Potency)"
    if mic_uM < 10:
        return "✅ Good Candidate"
    if mic_uM < 50:
        return "⚠️ Weak Candidate"
    return "❌ Inactive"


def main() -> None:
    st.set_page_config(
        page_title="💊 AI Peptide Dosing Calculator",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    model = load_model()

    # Sidebar
    st.sidebar.title("About the Model")
    st.sidebar.info("RandomForestRegressor\nR² = 0.45\nRMSE = 0.63 log units")

    # Header
    st.title("💊 AI Peptide Dosing Calculator")
    st.markdown("Predict antimicrobial peptide MIC from sequence features.")

    default_seq = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    sequence = st.text_area(
        "Paste amino acid sequence",
        value=default_seq,
        height=140,
        placeholder="Enter peptide sequence (e.g., LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES)",
    )
    analyze = st.button("Analyze", type="primary")

    if analyze:
        if model is None:
            st.stop()

        props = get_properties(sequence)
        feature_df = pd.DataFrame([props])

        # Align columns with training
        if hasattr(model, "feature_names_in_"):
            feature_df = feature_df.reindex(columns=model.feature_names_in_, fill_value=0)

        y_pred = float(model.predict(feature_df)[0])  # predicted -log10(MIC)
        mic_uM = float(10 ** (-y_pred))

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.metric("Predicted MIC (µM)", f"{mic_uM:.3f}")
            st.write(interpret_mic(mic_uM))
        with col2:
            st.metric("Predicted -log10(MIC)", f"{y_pred:.3f}")

        st.markdown("### Physicochemical Properties")
        st.dataframe(pd.DataFrame([props]).T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
