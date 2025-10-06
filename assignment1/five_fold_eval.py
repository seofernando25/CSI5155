import json
from tqdm import tqdm
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from imblearn.metrics import specificity_score
from pipeline import load_dataframe, split_features, build_preprocessor, build_pipeline
from models import get_model_module, MODELS


def evaluate_model_5fold(model_name, sampling, best_params):
    df = load_dataframe()
    X, y, num_cols, cat_cols = split_features(df)

    model_module = get_model_module(model_name)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    clf = model_module.get_model()
    pipeline = build_pipeline(preprocessor, clf, sampling=sampling)
    pipeline.set_params(**best_params)
    y_pred_proba = cross_val_predict(
        pipeline,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    specificity = specificity_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)
    cm = confusion_matrix(y, y_pred)

    return {
        "model": model_name,
        "sampling": sampling,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }


def main():
    with open("tuning_results.json", "r") as f:
        tuning_results = json.load(f)

    best_combinations = {}

    for model in MODELS:
        best_recall = -1
        best_result = None
        for result in tuning_results:
            if result["model"] == model and result["metrics"]["recall"] > best_recall:
                best_recall = result["metrics"]["recall"]
                best_result = result
        if best_result:
            best_combinations[model] = best_result

    five_fold_results = []
    for model, result in tqdm(
        best_combinations.items(), total=len(best_combinations), desc="5-fold eval"
    ):
        model_name = result["model"]
        sampling = result["sampling"]
        best_params = result["best_params"]

        try:
            eval_result = evaluate_model_5fold(model_name, sampling, best_params)
            five_fold_results.append(eval_result)
        except Exception as e:
            tqdm.write(f"Error evaluating {model_name}[{sampling}]: {e}")

    with open("five_fold_results.json", "w") as f:
        json.dump(five_fold_results, f, indent=2)


if __name__ == "__main__":
    main()
