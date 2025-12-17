import joblib

model = joblib.load('projects/MIC Regression/models/mic_predictor.pkl')
print(f'Total features: {len(model.feature_names_in_)}')

kmer_features = [f for f in model.feature_names_in_ if f.startswith('kmer_')]
print(f'K-mer features: {len(kmer_features)}')
print(f'First 10 k-mers: {kmer_features[:10]}')
print(f'Last 10 k-mers: {kmer_features[-10:]}')
