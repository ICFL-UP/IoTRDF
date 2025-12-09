"""
classifier_evaluator.py

Provides the ClassifierEvaluator utility used in the IRDF backend to train and
evaluate supervised classifiers on a selected feature subset. It wraps several
scikit-learn models (Random Forest, Decision Tree, Logistic Regression, SVM,
and KNN), computes accuracy / precision / recall / F1, and exposes
feature_importances_ aligned with the full state dimension.
"""
import numpy as np
import logging
import warnings
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from sklearn.exceptions import ConvergenceWarning

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)

# ---- Supported classifier names (UI should pass one of these strings) ----
SUPPORTED_CLASSIFIERS = {"rf", "dt", "svm", "logreg", "knn"}

@dataclass
class ClassifierSpec:
    name: str = "rf"
    params: Dict = field(default_factory=dict)

def build_classifier(spec: ClassifierSpec, n_jobs_default: int = -1, random_state_default: int = 42):
    """Return a sklearn estimator based on spec.name and spec.params."""
    name = spec.name.lower()
    params = dict(spec.params or {})

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth"),
            class_weight=params.get("class_weight", "balanced"),
            random_state=params.get("random_state", random_state_default),
            n_jobs=params.get("n_jobs", n_jobs_default),
        )
    if name == "dt":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth"),
            class_weight=params.get("class_weight", "balanced"),
            random_state=params.get("random_state", random_state_default),
        )
    if name == "logreg":
        
        return LogisticRegression(
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "saga"),
            max_iter=params.get("max_iter", 1000), 
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=params.get("n_jobs", n_jobs_default),
            random_state=params.get("random_state", random_state_default),
        )
    if name == "svm":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            class_weight=params.get("class_weight", "balanced"),
            probability=params.get("probability", False),
            random_state=params.get("random_state", random_state_default),
        )
    if name == "knn":
        return KNeighborsClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            p=params.get("p", 2),
            n_jobs=params.get("n_jobs", n_jobs_default),
        )

    raise ValueError(f"Unsupported classifier: {spec.name}. Supported: {sorted(SUPPORTED_CLASSIFIERS)}")


class ClassifierEvaluator:
    def __init__(self, state_dim: int, classifier_spec: Optional[ClassifierSpec] = None):
        self.state_dim = state_dim
        self.spec = classifier_spec or ClassifierSpec(name="rf", params={})
        self.clf = build_classifier(self.spec)
        self.feature_importances_ = np.zeros(state_dim, dtype=np.float32)
        self.trained = False
        logger.info(f"[ClassifierEvaluator] Initialized with {state_dim} features using '{self.spec.name}'.")

    # ---- public API ----

    def set_classifier(self, classifier_spec: ClassifierSpec):
        """Switch classifier on the fly (useful if UI changes selection between runs)."""
        self.spec = classifier_spec
        self.clf = build_classifier(self.spec)
        self.trained = False
        self.feature_importances_.fill(0)

    def train(self, X: np.ndarray, y: np.ndarray, feature_mask: np.ndarray) -> Dict[str, float]:
        try:
            feature_mask = np.asarray(feature_mask).reshape(-1)
            if feature_mask.shape[0] != self.state_dim:
                raise ValueError(f"[train] Feature mask has wrong shape: expected ({self.state_dim},) but got {feature_mask.shape}")

            selected_features_mask = feature_mask.astype(bool)
            num_selected = int(selected_features_mask.sum())
            if num_selected == 0:
                logger.warning("[train] Empty feature mask received, returning empty metrics.")
                return self._empty_metrics()

            X_selected = X[:, selected_features_mask]
            y_int = self._validate_labels(y)
            
            # --- FIX: Temporarily suppress warnings for cleaner output ---
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                
                # Cross-val predictions with current classifier
                preds = cross_val_predict(self.clf, X_selected, y_int, cv=3)
                metrics = self._compute_metrics(y_int, preds)

                # Fit on full (selected) training set
                self.clf.fit(X_selected, y_int)
                self.trained = True

            # Update importances (robust across model types)
            self._update_feature_importances(selected_features_mask)

            logger.info(f"[train] ({self.spec.name}) Trained with {num_selected} features — {self._format_metrics(metrics)}")
            return metrics

        except Exception as e:
            logger.error(f"[train] Training failed: {str(e)}", exc_info=True)
            return self._empty_metrics()

    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            y_int = self._validate_labels(y)
           
            if len(np.unique(y_int)) < 2:
                logger.warning("[train_on_batch] Skipping batch — only one class present.")
                return 1.0
            if X.ndim == 1: X = X.reshape(-1, 1)
            
          
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                
                self.clf.fit(X, y_int)
                self.trained = True
            
            preds = self.clf.predict(X)
            acc = accuracy_score(y_int, preds)
            return 1.0 - acc
        except Exception as e:
            logger.error(f"[train_on_batch] Batch training failed: {str(e)}", exc_info=True)
            return 1.0

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_mask: np.ndarray) -> Dict[str, float]:
        try:
            if not self.trained:
                raise RuntimeError("Classifier not trained before evaluation. Call .train() or .train_on_batch() first.")
            selected = feature_mask.astype(bool)
            if selected.sum() == 0:
                return self._empty_metrics()
            X_sel = X[:, selected]
            if X_sel.ndim == 1: X_sel = X_sel.reshape(-1, 1)
            
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                preds = self.clf.predict(X_sel)
            
            return self._compute_metrics(y, preds)
        except Exception as e:
            logger.error(f"[evaluate] Evaluation failed: {str(e)}", exc_info=True)
            return self._empty_metrics()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Classifier must be trained before calling predict()")
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            return self.clf.predict(X)

    # ---- helpers ----

    def _update_feature_importances(self, selected_mask: np.ndarray):
        n_selected = int(selected_mask.sum())
        importances_subset = None

        # 1) Tree-based models expose .feature_importances_
        if hasattr(self.clf, "feature_importances_"):
            importances_subset = np.asarray(self.clf.feature_importances_, dtype=np.float32)

        # 2) Linear models: use |coef| (average across classes if multiclass)
        elif hasattr(self.clf, "coef_"):
            coef = np.asarray(self.clf.coef_, dtype=np.float32)
            if coef.ndim == 1:
                importances_subset = np.abs(coef)
            else:
                importances_subset = np.abs(coef).mean(axis=0)
            importances_subset = importances_subset.astype(np.float32)

        # 3) Otherwise: fill zeros (fast, safe default)
        if importances_subset is None or importances_subset.shape[0] != n_selected:
            importances_subset = np.zeros(n_selected, dtype=np.float32)

        new_full = np.zeros(self.state_dim, dtype=np.float32)
        new_full[selected_mask] = importances_subset
        self.feature_importances_ = new_full

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = self._validate_labels(y_true)
        unique_classes = np.unique(y_true)
        avg = 'binary' if len(unique_classes) == 2 else 'weighted'
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        }

    def _validate_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1)
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(int)
        return y

    def _empty_metrics(self) -> Dict[str, float]:
        return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def _format_metrics(self, metrics: Dict) -> str:
        return ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())