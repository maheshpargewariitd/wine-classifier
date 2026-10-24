import joblib
from pathlib import Path 
from sklearn.datasets import load_wine
project_root = Path(__file__).resolve().parents[1]
model_path = project_root/"models"/"wine_knn.joblib"
model = joblib.load(model_path)

wine = load_wine(as_frame=True)
sample = wine.data.iloc[[0]]

prediction = model.predict(sample)[0]

print(f"Predicted class ID: {prediction}")
print(f"Predicted class name: {wine.target_names[prediction]}")