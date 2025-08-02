from sklearn.metrics import roc_auc_score, classification_report, recall_score


def evaluate_model(name, y_true, preds, probs):
    """Print evaluation metrics for a given model."""
    print(f"\n{name} Results:")
    print("ROC-AUC:", roc_auc_score(y_true, probs))
    print("Recall:", recall_score(y_true, preds))
    print(classification_report(y_true, preds))
