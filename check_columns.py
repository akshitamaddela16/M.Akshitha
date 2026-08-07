import pandas as pd
import joblib

csv_path = 'Heart_Disease_Prediction.csv'
model_path = 'heart_disease_model.pkl'

print('Reading CSV header...')
csv_cols = list(pd.read_csv(csv_path, nrows=0).columns)
print('CSV columns:')
for c in csv_cols:
    print(' -', c)

print('\nLoading model...')
model = joblib.load(model_path)
model_cols = list(getattr(model, 'feature_names_in_', []))
print('Model feature_names_in_:')
for c in model_cols:
    print(' -', c)

csv_set = set(csv_cols)
model_set = set(model_cols)

print('\nColumns in model but not in CSV:')
for c in sorted(model_set - csv_set):
    print(' -', c)

print('\nColumns in CSV but not in model:')
for c in sorted(csv_set - model_set):
    print(' -', c)

# Helpful hint: excluding target column if present
possible_target = 'Heart Disease'
if possible_target in csv_set and possible_target not in model_set:
    print(f"\nNote: '{possible_target}' appears to be the target column in the CSV and is not expected by the model (that's normal).")
