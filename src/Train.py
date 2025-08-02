import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_data
from src.models import get_xgb_model, get_catboost_model, get_ffn_model
from src.evaluate import evaluate_model

DATA_PATH = 'data/drug_discovery_virtual_screening.csv'  # Update path as needed
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['active', 'compound_id', 'binding_affinity'])
y = df['active']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cat_cols = ['protein_id']
X_train, X_test, y_train = preprocess_data(X_train, X_test, y_train, cat_cols)

xgb_model = get_xgb_model()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

cat_model = get_catboost_model()
cat_model.fit(X_train, y_train)
cat_preds = cat_model.predict(X_test)
cat_probs = cat_model.predict_proba(X_test)[:, 1]

ffn_model = get_ffn_model(input_dim=X_train.shape[1])
ffn_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=0,
    validation_split=0.1
)
ffn_probs = ffn_model.predict(X_test).ravel()
ffn_preds = (ffn_probs >= 0.5).astype(int)
evaluate_model("XGBoost", y_test, xgb_preds, xgb_probs)
evaluate_model("CatBoost", y_test, cat_preds, cat_probs)
evaluate_model("FFN", y_test, ffn_preds, ffn_probs)

