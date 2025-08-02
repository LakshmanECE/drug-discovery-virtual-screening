# Drug Discovery Virtual Screening

## Overview

Machine learning and deep learning models (**XGBoost**, **CatBoost**, and **Feedforward Neural Network**) to predict **bioactive compounds** for drug discovery. The pipeline processes chemical-protein interaction data and classifies compounds as **active** or **inactive**, enabling efficient virtual screening.

## Dataset

* **Source**: [Kaggle Drug Discovery Virtual Screening Dataset](https://www.kaggle.com/datasets/USERNAME/drug-discovery-virtual-screening-dataset)
* **Features**: Compound descriptors, protein IDs, binding affinity
* **Target**: `active` (binary classification)

## Workflow

1. **Preprocessing**: Target encoding, KNN imputation, robust scaling
2. **Modeling**: XGBoost, CatBoost, Feedforward Neural Network (Keras)
3. **Evaluation**: ROC-AUC, Recall, F1-score comparison

## Results

| Model       | ROC-AUC | Recall | Accuracy |
| ----------- | ------- | ------ | -------- |
| XGBoost     | 0.90    | 0.59   | 84%      |
| CatBoost    | 0.92    | 0.68   | 86%      |
| FFN (Keras) | 0.94    | 0.71   | 88%      |

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, TensorFlow/Keras, Category Encoders

## How to Run

```bash
git clone https://github.com/<your-username>/drug-discovery-virtual-screening.git
cd drug-discovery-virtual-screening
pip install -r requirements.txt
python src/train_evaluate.py
```

## Future Improvements

* Hyperparameter tuning (Optuna)
* Integrate molecular fingerprints (RDKit)
* Deploy model as a web API for screening

## License

MIT License
