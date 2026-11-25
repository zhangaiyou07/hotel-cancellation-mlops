"""Model sanity tests: valid outputs and better than a naive baseline."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.features import TARGET, make_pipeline


def _split(sample_df):
    X, y = sample_df.drop(columns=[TARGET]), sample_df[TARGET]
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def test_predictions_are_valid_probabilities(sample_df):
    X_tr, X_te, y_tr, _ = _split(sample_df)
    pipe = make_pipeline(LogisticRegression(max_iter=2000), kind="onehot").fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_beats_naive_baseline(sample_df):
    X_tr, X_te, y_tr, y_te = _split(sample_df)
    naive = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    model = make_pipeline(LogisticRegression(max_iter=2000), kind="onehot").fit(X_tr, y_tr)
    naive_auc = roc_auc_score(y_te, naive.predict_proba(X_te)[:, 1])
    model_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    assert model_auc > naive_auc + 0.15, f"model AUC {model_auc:.3f} vs naive {naive_auc:.3f}"
