import joblib
import pandas as pd

model = joblib.load('heart_disease_model.pkl')
print('Model feature_names_in_:', getattr(model, 'feature_names_in_', None))

# Sample values matching app.py defaults
row = {
    'Age': 45,
    'Sex': 1,
    'Chest pain type': 2,  # ATA
    'BP': 120,
    'Cholesterol': 200,
    'FBS over 120': 0,
    'EKG results': 0,
    'Max HR': 150,
    'Exercise angina': 0,
    'ST depression': 1.0,
    'Slope of ST': 1,
    'Number of vessels fluro': 0,
    'Thallium': 3
}

cols = list(getattr(model, 'feature_names_in_', row.keys()))

df = pd.DataFrame([row], columns=cols)
print('Input dtypes:')
print(df.dtypes)

pred = model.predict(df)
proba = model.predict_proba(df)
print('Prediction:', pred[0])
print('Probability:', proba[0])
