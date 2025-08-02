from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow import keras
def get_xgb_model():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
def get_catboost_model():
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        verbose=0,
        random_state=42
    )

def get_ffn_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['AUC', 'Recall']
    )

    return model
