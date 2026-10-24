# Wine Classifier

This project predicts a wine class from its measured chemical features.

## Model

The project uses a scikit-learn pipeline with feature scaling and a KNN
classifier. Five-fold cross-validation is used to select the number of
neighbors.

## Run it

Activate the `campusx` environment, then run these commands from the project
folder:

```bat
python src\train.py
python src\predict.py
```

Training saves the model in the `models` folder. Run training before making
a prediction.

## Results

- Selected `k=11` using 5-fold cross-validation.
- Mean cross-validation accuracy: `97.22%`.
- Held-out test accuracy: `100.00%` with `random_state=42`.

## Limitations

This project uses a small dataset. Test accuracy is one result from one
train/test split, so it should not be treated as a guarantee of future
performance.