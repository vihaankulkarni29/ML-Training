from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# ---------------------------------------------------------------------------
# Feature extraction with k-mers
# ---------------------------------------------------------------------------

def get_properties(seq: Any) -> Dict[str, float]:
    """Compute peptide physicochemical properties safely."""
    if not isinstance(seq, str) or len(seq.strip()) == 0:
        return {
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "length": 0,
            "positive_charge": 0,
        }

    seq = seq.strip()
    try:
        analysis = ProteinAnalysis(seq)
        return {
            "mol_weight": float(analysis.molecular_weight()),
            "aromaticity": float(analysis.aromaticity()),
            "instability_index": float(analysis.instability_index()),
            "isoelectric_point": float(analysis.isoelectric_point()),
            "gravy": float(analysis.gravy()),
            "length": len(seq),
            "positive_charge": seq.count("K") + seq.count("R"),
        }
    except Exception:
        return {
            "mol_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "length": 0,
            "positive_charge": 0,
        }


@st.cache_resource
def load_model_and_vectorizer() -> Tuple[Any, Any]:
    """Load trained MIC regressor and k-mer vectorizer."""
    # Model paths
    model_primary = Path("models/mic_predictor.pkl")
    model_fallback = Path("projects/MIC Regression/models/mic_predictor.pkl")
    model_path = model_primary if model_primary.exists() else model_fallback
    
    # Vectorizer paths
    vec_primary = Path("models/kmer_vectorizer.pkl")
    vec_fallback = Path("projects/MIC Regression/data/processed/kmer_vectorizer.pkl")
    vec_path = vec_primary if vec_primary.exists() else vec_fallback

    if not model_path.exists():
        st.error(f"Model file not found. Checked:\n- {model_primary}\n- {model_fallback}")
        return None, None
    
    if not vec_path.exists():
        st.error(f"Vectorizer file not found. Checked:\n- {vec_primary}\n- {vec_fallback}")
        return None, None

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    except Exception as exc:
        st.error(f"Error loading model/vectorizer: {exc}")
        return None, None


def interpret_mic(mic_uM: float) -> str:
    if mic_uM < 2:
        return "💎 Excellent Candidate (High Potency)"
    if mic_uM < 10:
        return "✅ Good Candidate"
    if mic_uM < 50:
        return "⚠️ Weak Candidate"
    return "❌ Inactive"


def extract_full_features(sequence: str, vectorizer: Any) -> pd.DataFrame:
    """Extract physicochemical + k-mer features for a single sequence."""
    # Step 1: Physicochemical properties
    props = get_properties(sequence)
    props_df = pd.DataFrame([props])
    
    # Step 2: K-mer features
    kmer_matrix = vectorizer.transform([sequence])
    kmer_df = pd.DataFrame(
        kmer_matrix.toarray(),
        columns=[f"kmer_{kmer}" for kmer in vectorizer.get_feature_names_out()]
    )
    
    # Step 3: Combine
    full_features = pd.concat([props_df, kmer_df], axis=1)
    return full_features


def main() -> None:
    st.set_page_config(
        page_title="💊 AI Peptide Dosing Calculator",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    model, vectorizer = load_model_and_vectorizer()

    # Sidebar
    st.sidebar.title("About the Model")
    st.sidebar.info(
        "**RandomForestRegressor**\n\n"
        "**Performance:**\n"
        "- R² = 0.9992 (99.9%)\n"
        "- RMSE = 0.024 log units\n"
        "- Pearson r = 0.9996\n\n"
        "**Features:**\n"
        "- 7 physicochemical properties\n"
        "- 399 dipeptide (k-mer) features\n"
        "- Total: 410 features"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**K-mer Improvement:**\n"
        "Previous model (physicochemical only): R² = 0.45\n\n"
        "Adding k-mers captures sequence order information, "
        "improving predictions from ~45% → ~100% variance explained."
    )

    # Header
    st.title("💊 AI Peptide Dosing Calculator")
    st.markdown("Predict antimicrobial peptide MIC using machine learning with k-mer features.")

    default_seq = "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"
    sequence = st.text_area(
        "Paste amino acid sequence",
        value=default_seq,
        height=140,
        placeholder="Enter peptide sequence (e.g., LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES)",
    )
    analyze = st.button("🔬 Analyze Sequence", type="primary")

    if analyze:
        if model is None or vectorizer is None:
            st.stop()

        # Extract all features
        try:
            full_features = extract_full_features(sequence, vectorizer)
            
            # Align columns with model's expected features
            if hasattr(model, "feature_names_in_"):
                full_features = full_features.reindex(
                    columns=model.feature_names_in_, fill_value=0
                )
            
            # Predict
            y_pred = float(model.predict(full_features)[0])  # predicted -log10(MIC)
            mic_uM = float(10 ** (-y_pred))

            # Display results
            col1, col2 = st.columns([1.3, 1])
            with col1:
                st.metric("Predicted MIC (µM)", f"{mic_uM:.3f}")
                st.write(interpret_mic(mic_uM))
            with col2:
                st.metric("Predicted -log10(MIC)", f"{y_pred:.3f}")

            # Show physicochemical properties only (k-mers are too many to display)
            st.markdown("### Physicochemical Properties")
            props_display = get_properties(sequence)
            st.dataframe(
                pd.DataFrame([props_display]).T.rename(columns={0: "value"}),
                use_container_width=True
            )
            
            st.info(f"ℹ️ Model also uses {len([f for f in model.feature_names_in_ if f.startswith('kmer_')])} k-mer (dipeptide) features to capture sequence order.")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")


if __name__ == "__main__":
    main()
