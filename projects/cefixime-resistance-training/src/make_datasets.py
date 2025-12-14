import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os

def create_ml_dataset(input_file, target_antibiotic, output_filename):
    print(f"--- Processing for Target: {target_antibiotic} ---")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found.")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except:
        df = pd.read_csv(input_file, sep='\t', low_memory=False)
        
    # 2. Filter for rows where the antibiotic was actually tested
    # We look for the string "antibiotic=" in the AST phenotypes column
    df_clean = df.dropna(subset=['AST phenotypes'])
    mask = df_clean['AST phenotypes'].str.contains(target_antibiotic, case=False, na=False)
    df_target = df_clean[mask].copy()
    
    if len(df_target) == 0:
        print(f"ERROR: No samples found for {target_antibiotic}")
        return

    # 3. Parse the Target (Y)
    def get_label(pheno_str):
        # Example str: "amikacin=S,ceftriaxone=R,gentamicin=S"
        parts = pheno_str.split(',')
        for p in parts:
            if target_antibiotic.lower() in p.lower():
                try:
                    status = p.split('=')[1].strip()
                    if status in ['R', 'Resistant']:
                        return 1
                    elif status in ['S', 'Susceptible']:
                        return 0
                except IndexError:
                    continue
        return None # Intermediate or Unknown

    df_target['target'] = df_target['AST phenotypes'].apply(get_label)
    df_final = df_target.dropna(subset=['target']) # Drop Intermediates

    # 4. Parse Features (X) - One Hot Encoding Genes
    # Clean the gene string: "blaCTX-M-15=COMPLETE" -> "blaCTX-M-15"
    def clean_genes(gene_str):
        if pd.isna(gene_str): return []
        # Remove '=COMPLETE', '=PARTIAL', etc.
        genes = [g.split('=')[0].strip() for g in gene_str.split(',')]
        return genes

    df_final['gene_list'] = df_final['AMR genotypes'].apply(clean_genes)

    # Convert list of strings to Binary Matrix (0s and 1s)
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df_final['gene_list'])
    feature_names = mlb.classes_
    
    # Create the final DataFrame
    df_ml = pd.DataFrame(X, columns=feature_names, index=df_final['BioSample'])
    df_ml['resistance_label'] = df_final['target'].values
    
    # 5. Save
    print(f"Samples: {len(df_ml)}")
    print(f"Features: {len(feature_names)}")
    print(f"Resistance Rate: {df_ml['resistance_label'].mean():.2%}")
    df_ml.to_csv(output_filename)
    print(f"Saved -> {output_filename}\n")

# --- EXECUTE ---
if __name__ == "__main__":
    INPUT_FILE = 'data/raw/Ciproflaxcin_Cefi.csv'
    
    # Project 1: Cefixime Resistance (primary)
    create_ml_dataset(INPUT_FILE, 'cefixime', 'data/processed/dataset_cefixime.csv')
    
    # Project 2: Ceftriaxone (secondary)
    create_ml_dataset(INPUT_FILE, 'ceftriaxone', 'data/processed/dataset_ceftriaxone.csv')
    
    # Project 3: Chloramphenicol (tertiary)
    create_ml_dataset(INPUT_FILE, 'chloramphenicol', 'data/processed/dataset_chloramphenicol.csv')
