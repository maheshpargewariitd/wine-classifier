from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


def load_data():
    wine = load_wine(as_frame=True)
    df = wine.frame

    X = df.drop(columns="target")
    y = df["target"]

    return X, y


def build_model(n_neighbors):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])

def select_best_k(X_train, y_train, candidates=(3, 5, 7, 9, 11), cv=5):
    scores_by_k = {}

    for k in candidates:
        candidate_model = build_model(n_neighbors=k)
        scores = cross_val_score(
            candidate_model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
        )
        scores_by_k[k] = scores.mean()

    best_k = max(scores_by_k, key=scores_by_k.get)
    return best_k, scores_by_k
def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    best_k, cv_scores = select_best_k(X_train, y_train)

    for k, score in cv_scores.items():
        print(f"k={k}: cross-validation accuracy={score:.2%}")

    print(f"Selected k={best_k}")

    model = build_model(n_neighbors=best_k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Final test accuracy: {accuracy:.2%}")
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "wine_knn.joblib"
    joblib.dump(model, model_path)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()