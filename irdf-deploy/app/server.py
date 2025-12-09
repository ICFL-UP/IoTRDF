import os
import time
import json
from typing import List, Optional, Tuple

import sys
import types

import joblib
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline # Explicitly import Pipeline for type checking

# -------------------------------------------------------------------
# FeatureSelector definition (same name as in your training code)
# -------------------------------------------------------------------
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that selects a subset of columns.
    """

    def __init__(self, indices=None):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.indices is None:
            return X
        idx = np.asarray(self.indices, dtype=int)
        
        # Ensure X is 2D for consistent slicing behavior
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Slice based on indices
        # If indices is a list of single element (e.g., [82]), this returns a 2D array: (n_samples, 1)
        return X[:, idx]


# Make FeatureSelector available as __main__.FeatureSelector (required for joblib)
main_mod = sys.modules.get("__main__")
if main_mod is None:
    main_mod = types.ModuleType("__main__")
    sys.modules["__main__"] = main_mod
setattr(main_mod, "FeatureSelector", FeatureSelector)
# -------------------------------------------------------------------

# ---------- Config ----------
DEFAULT_SET = os.environ.get("IRDF_MODEL_SET", "policy").lower()

OVERRIDE_MODEL_PATH = os.environ.get("IRDF_MODEL_PATH")
OVERRIDE_FEATURES_PATH = os.environ.get("IRDF_FEATURES_PATH")

MODELS_ROOT = "/models"
# ----------------------------

app = FastAPI(title="IRDF Inference Service")


class Item(BaseModel):
    x: List[float]  # raw 90-dim vector


# Globals
_model = None # This will now store EITHER the Pipeline or the raw DT estimator
_selected: Optional[List[int]] = None
_active_set: Optional[str] = None

_meta: dict = {}
_expected_dim: Optional[int] = None
_total_dim: Optional[int] = None


# ---------- Helpers ----------
def _set_paths(set_name: str) -> Tuple[str, Optional[str]]:
    if OVERRIDE_MODEL_PATH:
        return OVERRIDE_MODEL_PATH, OVERRIDE_FEATURES_PATH

    folder = os.path.join(MODELS_ROOT, set_name)
    model_path = os.path.join(folder, "model.joblib")
    meta_path = os.path.join(folder, "metadata.json")
    alt_path = os.path.join(MODELS_ROOT, "selected_features.json")

    if os.path.exists(meta_path):
        return model_path, meta_path
    if os.path.exists(alt_path):
        return model_path, alt_path
    return model_path, None


def _read_meta(features_path: Optional[str]):
    if not features_path:
        return None, None, None, {}
    p = Path(features_path)
    if not p.exists():
        return None, None, None, {}

    try:
        data = json.loads(p.read_text())

        if isinstance(data, dict):
            fi = data.get("feature_indices", None)
            if fi is None or fi == "ALL":
                sel = None
            elif isinstance(fi, list):
                sel = [int(i) for i in fi]
            else:
                sel = None

            expected = data.get("expected_input_dim", None)
            total = data.get("n_features_total", None)
            return sel, expected, total, data

        if isinstance(data, list):
            sel = [int(i) for i in data]
            return sel, len(sel), None, {"feature_indices": sel}

    except Exception:
        pass

    return None, None, None, {}


def _load_model(set_name: str):
    global _model, _selected, _active_set, _meta, _expected_dim, _total_dim

    model_path, features_path = _set_paths(set_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load the object (Pipeline or raw estimator)
    m = joblib.load(model_path)
    sel, exp_dim, total_dim, meta = _read_meta(features_path)

    # --- OPTIMIZATION LOGIC FOR POLICY MODEL ---
    # Check if this is the Policy model, is saved as a Pipeline, expects 1 input feature, and we know which one.
    if set_name == "policy" and isinstance(m, Pipeline) and exp_dim == 1 and sel and len(sel) == 1:
        # Policy Optimization: Extract the raw Decision Tree estimator (the last step)
        _model = m.steps[-1][1]
        print(f"[IRDF OPT] Policy loaded raw: Bypassing FeatureSelector for index {sel[0]}.")
    else:
        # For Baseline or any other model, load the full Pipeline object
        _model = m

    _selected = sel
    _active_set = set_name
    _meta = meta
    _expected_dim = exp_dim
    _total_dim = total_dim


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------- Lifecycle ----------
@app.on_event("startup")
def _startup():
    try:
        _load_model(DEFAULT_SET)
    except Exception:
        alt = "baseline" if DEFAULT_SET == "policy" else "policy"
        _load_model(alt)

    print(f"[IRDF] Loaded set='{_active_set}', selected={_selected}")


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "service": "IRDF Inference",
        "health": "/health",
        "predict": "/predict",
        "info": "/model/info",
    }


@app.get("/health")
def health():
    p = psutil.Process()
    mem = p.memory_info().rss
    cpu = p.cpu_times()
    return {
        "status": "ok",
        "rss_bytes": mem,
        "cpu_user": cpu.user,
        "cpu_system": cpu.system,
        "active_set": _active_set,
    }


@app.get("/model/info")
def model_info():
    model_path, _ = _set_paths(_active_set or DEFAULT_SET)
    return {
        "active_set": _active_set,
        "selected_indices": _selected,
        "model_path": model_path,
        "meta": _meta,
    }


@app.post("/model/reload")
def reload_model(set: str = Query("policy", pattern="^(policy|baseline)$")):
    try:
        _load_model(set)
        return {"status": "ok", "active_set": _active_set, "selected_indices": _selected}
    except Exception as e:
        raise HTTPException(400, f"Reload failed: {e}")


@app.post("/predict")
def predict(item: Item):
    if _model is None:
        raise HTTPException(500, "Model not loaded")

    t0_wall = time.perf_counter()
    proc = psutil.Process()
    cpu0 = proc.cpu_times()
    rss0 = proc.memory_info().rss

    x_full = np.asarray(item.x, dtype=float)

    # --------------------------------------------------------------------
    # OPTIMIZATION INFERENCE LOGIC: Bypasses the FeatureSelector step
    # --------------------------------------------------------------------
    if _active_set == "policy" and _expected_dim == 1 and _selected and len(_selected) == 1:
        # Manual slicing for the Policy DT, which expects only 1 feature
        
        # Get the feature index (e.g., 82)
        feature_index = _selected[0] 
        
        # Extract the single feature and reshape it into a 2D array for predict([[value]])
        x_processed = np.array([x_full[feature_index]]).reshape(1, -1)
        model_to_use = _model # This is the raw DT estimator
        
    else:
        # For Baseline (or any other Pipeline model), use the full pipeline path.
        # The internal FeatureSelector will execute here.
        x_processed = [x_full]
        model_to_use = _model

    try:
        # Use the determined model and processed input
        pred = model_to_use.predict(x_processed)[0]
        try:
            # Predict_proba should also use the processed input
            proba = float(model_to_use.predict_proba(x_processed)[0][1])
        except Exception:
            proba = None
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")

    cpu1 = proc.cpu_times()
    rss1 = proc.memory_info().rss
    t1_wall = time.perf_counter()

    return {
        "pred": int(pred),
        "proba": proba,
        "latency_ms": (t1_wall - t0_wall) * 1000.0,
        "proc_cpu_s": (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system),
        "rss_bytes": rss1,
        "peak_rss_bytes": max(rss0, rss1),
        "ts_ms": now_ms(),
        "active_set": _active_set,
    }
