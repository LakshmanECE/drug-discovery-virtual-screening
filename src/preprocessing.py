import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


def preprocess_data(X_train, X_test, y_train, cat_cols):
    # Target Encoding
    encoder = TargetEncoder(cols=cat_cols, smoothing=10)
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols], y_train)
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Scaling
    scaler = RobustScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Convert to numpy
    return (
        np.array(X_train, dtype=np.float32),
        np.array(X_test, dtype=np.float32),
        np.array(y_train, dtype=np.int32)
    )
