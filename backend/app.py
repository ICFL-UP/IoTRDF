"""
IoT Ransomware Detection API Server
Main Flask application for dataset preprocessing, feature selection via RL,
classifier training/evaluation, and model export for Docker deployment
Provides full pipeline from raw data to production-ready models
"""
import os
import sys
import json
import traceback
import time
import logging
import threading
import io
import csv
import pickle
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, Response, send_file, make_response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
from mysql.connector import connect, Error
import joblib
from joblib import dump, load as joblib_load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import custom modules
from modules.data_clearning import DataCleaner
from modules.data_splitter import DataSplitter
from modules.classifier_evaluator import ClassifierEvaluator, ClassifierSpec, build_classifier
from modules.data_storage import save_preprocessed_splits
from modules.policy_learner import PolicyLearner



app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAINING_IN_PROGRESS = False
should_stop_training = threading.Event()

validation_progress = {}
validation_logs = {}

# Directory paths
DATASET_OUTPUT_DIR = Path(r"C:\Users\Mohlale-PC\Desktop\Best_Version\Frontend\Preprocessing_datasets")

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "0712252511",
    "database": "iotransomwaredb"
}


class FeatureSelector(BaseEstimator, TransformerMixin):
    """A simple sklearn-compatible feature selector."""
    def __init__(self, indices):
        self.indices = indices
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        result = X[:, self.indices]
        if result.ndim == 1:
            return result.reshape(-1, 1)
        return result
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def get_db_connection():
    """Establish database connection."""
    try:
        connection = connect(**DB_CONFIG)
        logger.info("Database connection successful.")
        return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        raise

def safe_torch_mean(value):
    """Safe calculation of mean for various input types."""
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.mean().item())
    if isinstance(value, (np.ndarray, list)):
        return float(np.mean(value))
    if isinstance(value, (float, int)):
        return float(value)
    return 0.0

def pad_action(action: List[int], max_len: int = 3) -> List[int]:
    """Pad or truncate action list to fixed length."""
    return (action + [0] * max_len)[:max_len]

def safe_convert_data(data):
    """Safely convert and validate input data with numerical checks."""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, (list, np.ndarray)):
            arr = np.array(data, dtype=np.float32)
            return np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
        return data
    except Exception as e:
        logger.warning(f"Data conversion warning: {str(e)}")
        return np.zeros(1, dtype=np.float32)

def parse_action_string(action_str: str) -> List[int]:
   
    try:
        action_type, features_str = action_str.split(":")
        features = list(map(int, features_str.split(",")))
        return [int(action_type)] + features
    except Exception as e:
        logger.warning(f"Failed to parse action string '{action_str}': {e}")
        return [0, 0]

def normalize_rewards(rewards):
   
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    if max_reward > min_reward:
        normalized_rewards = (rewards - min_reward) / (max_reward - min_reward) * 2 - 1
    else:
        normalized_rewards = np.zeros_like(rewards)
    return normalized_rewards

def standardize_rewards(rewards):
   
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    if std_reward > 0:
        standardized_rewards = (rewards - mean_reward) / std_reward
    else:
        standardized_rewards = np.zeros_like(rewards)
    return standardized_rewards

def tensor_to_serializable(obj: Any) -> Any:
   
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple, set)):
        return [tensor_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: tensor_to_serializable(v) for k, v in obj.items()}
    if hasattr(obj, 'name'):
        return obj.name
    return obj

def load_data_split(dataset_id: int, split_name: str):

    if split_name not in ['train', 'val', 'test']:
        raise ValueError("split_name must be 'train', 'val', or 'test'")
    
    table_name = f"{split_name}ing_data"
    if split_name == 'val':
        table_name = "validation_data"
    x_col, y_col = f"X_{split_name}", f"y_{split_name}"

    db_conn = None
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        query = f"SELECT {x_col}, {y_col} FROM {table_name} WHERE dataset_id = %s"
        cursor.execute(query, (dataset_id,))
        row = cursor.fetchone()
        if row:
            X = np.array(json.loads(row[x_col]))
            y = np.array(json.loads(row[y_col]))
            return X, y
        return None, None
    finally:
        if db_conn and db_conn.is_connected():
            cursor.close()
            db_conn.close()

def load_training_data(dataset_id: int, include_test: bool = False) -> Tuple[np.ndarray, ...]:
   
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    try:
       
        cursor.execute("""
            SELECT X_train, y_train 
            FROM training_data 
            WHERE dataset_id = %s
        """, (dataset_id,))
        train_row = cursor.fetchone()
        
        cursor.execute("""
            SELECT X_val, y_val 
            FROM validation_data 
            WHERE dataset_id = %s
        """, (dataset_id,))
        val_row = cursor.fetchone()

        if not train_row or not val_row:
            raise ValueError(f"No training or validation data found for dataset_id {dataset_id}")

        X_train = np.array(json.loads(train_row['X_train']), dtype=np.float32)
        y_train = np.array(json.loads(train_row['y_train']), dtype=np.int64)
        X_val = np.array(json.loads(val_row['X_val']), dtype=np.float32)
        y_val = np.array(json.loads(val_row['y_val']), dtype=np.int64)

        logger.info(f" Loaded classification data — Train: {X_train.shape}, Val: {X_val.shape}")

        if include_test:
            cursor.execute("""
                SELECT X_test, y_test 
                FROM testing_data 
                WHERE dataset_id = %s
            """, (dataset_id,))
            test_row = cursor.fetchone()

            if not test_row:
                raise ValueError(f"No test data found for dataset_id {dataset_id}")

            X_test = np.array(json.loads(test_row['X_test']), dtype=np.float32)
            y_test = np.array(json.loads(test_row['y_test']), dtype=np.int64)

            logger.info(f"📊 Test set loaded: {X_test.shape}")
            return X_train, y_train, X_val, y_val, X_test, y_test

        return X_train, y_train, X_val, y_val

    except Exception as e:
        logger.error(f" Data loading failed: {str(e)}", exc_info=True)
        raise
    finally:
        cursor.close()
        db.close()

def build_inference_pipeline(classifier_spec, feature_indices, use_scaler=False):
    
    if not feature_indices or (isinstance(feature_indices, list) and len(feature_indices) == 0):
        steps = []
        if use_scaler:
            steps.append(("scale", StandardScaler()))
        steps.append(("clf", build_classifier(classifier_spec)))
        return Pipeline(steps)
    
    steps = [("select", FeatureSelector(feature_indices))]
    
    if use_scaler:
        steps.append(("scale", StandardScaler()))
    
    steps.append(("clf", build_classifier(classifier_spec)))
    return Pipeline(steps)


@app.route('/upload', methods=['POST'])
def upload_dataset():
    """Upload a CSV dataset."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    datasets_dir = os.path.join(os.getcwd(), 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)
    file_path = os.path.join(datasets_dir, secure_filename(file.filename))

    try:
    
        with open(file_path, 'wb') as f:
            file.save(f)
        
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO datasets (filename, file_path) VALUES (%s, %s)",
            (file.filename, file_path)
        )
        db.commit()
        
        return jsonify({
            "message": "File uploaded successfully",
            "dataset_id": cursor.lastrowid,
            "features": headers
        }), 200
        
    except csv.Error:
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": "Invalid CSV file"}), 400
    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'db' in locals(): db.close()

@app.route('/datasets', methods=['GET'])
def get_datasets():
    """Get list of uploaded datasets."""
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, filename FROM datasets")
        datasets = cursor.fetchall()
    except Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500
    finally:
        cursor.close()
        db.close()

    return jsonify({"datasets": datasets}), 200

@app.route('/preprocess', methods=['POST'])
def preprocess_dataset():
    
    payload = request.get_json(silent=True) or {}
    dataset_id = (
        payload.get('dataset_id')
        or payload.get('datasetId')
        or request.args.get('dataset_id')
        or request.args.get('datasetId')
    )
    if dataset_id in (None, "", "null", "undefined"):
        return jsonify({"error": "dataset_id is required"}), 400
    try:
        dataset_id = int(dataset_id)
    except ValueError:
        return jsonify({"error": "dataset_id must be an integer"}), 400

    db = None
    cursor = None

    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        logger.info(f"Starting preprocessing for dataset_id={dataset_id}")

        logger.info("Step 1: Cleaning raw data...")
        data_cleaner = DataCleaner(db, dataset_id=dataset_id)
        cleaned_data = data_cleaner.load_and_clean()
        if "error" in cleaned_data:
            logger.error(f"Data Cleaning Failed: {cleaned_data['error']}")
            return jsonify({"error": cleaned_data["error"]}), 500
        logger.info(" Data cleaning completed.")

        logger.info(" Step 2: Splitting data into train/val/test...")
        data_splitter = DataSplitter(cleaned_data, dataset_id=dataset_id, cursor=cursor)
        split_data = data_splitter.split_dataset(persist=False)
        if "error" in split_data:
            logger.error(f"Data Splitting Failed: {split_data['error']}")
            return jsonify({"error": split_data["error"]}), 500
        logger.info(" Data splitting completed.")

        logger.info(" Step 3: Saving preprocessed splits (DB + CSV)...")
        for split in ["train", "val", "test"]:
            X = split_data[f"X_{split}"]
            y = split_data[f"y_{split}"]
            save_preprocessed_splits(cursor, dataset_id, split, X, y)

        db.commit()
        logger.info(" Preprocessing completed successfully.")

        return jsonify({
            "message": "Preprocessing completed successfully",
            "X_train_shape": f"{split_data['X_train'].shape[0]} samples × {split_data['X_train'].shape[1]} features",
            "X_val_shape":   f"{split_data['X_val'].shape[0]} samples × {split_data['X_val'].shape[1]} features",
            "X_test_shape":  f"{split_data['X_test'].shape[0]} samples × {split_data['X_test'].shape[1]} features",
            "y_train_shape": int(split_data["y_train"].shape[0]),
            "y_val_shape":   int(split_data["y_val"].shape[0]),
            "y_test_shape":   int(split_data["y_test"].shape[0]),
            "class_weights": split_data.get("class_weights"),
            "logs":          split_data.get("logs", [])
        }), 200

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        if db:
            db.rollback()
        return jsonify({
            "error": f"Error during preprocessing: {e}",
            "traceback": traceback.format_exc()
        }), 500
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

def serve_preprocessed_file(dataset_id, dataset_type):
    """Serve preprocessed files for download."""
    file_mapping = {
        "train_features": f"{dataset_id}_train_X",
        "train_labels": f"{dataset_id}_train_y",
        "val_features": f"{dataset_id}_val_X",
        "val_labels": f"{dataset_id}_val_y",
        "test_features": f"{dataset_id}_test_X",
        "test_labels": f"{dataset_id}_test_y"
    }

    if dataset_type not in file_mapping:
        return jsonify({
            "error": "Invalid dataset type",
            "valid_types": list(file_mapping.keys())
        }), 400

    for ext in ['.csv']:
        candidate = DATASET_OUTPUT_DIR / f"{file_mapping[dataset_type]}{ext}"
        if candidate.exists():
            return send_file(
                str(candidate),
                as_attachment=True,
                download_name=candidate.name,
                mimetype='text/csv'
            )

    available = [f.name for f in DATASET_OUTPUT_DIR.glob(f"{dataset_id}_*.*")]
    return jsonify({
        "error": "Requested file not found",
        "searched_for": f"{file_mapping[dataset_type]}.csv",
        "available_files": available
    }), 404

@app.route('/preprocessing_results/<int:dataset_id>', methods=['GET'])
def get_preprocessing_results(dataset_id):
    """Get preprocessing results for a dataset."""
    db = None
    cursor = None
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        download_requested = request.args.get('download_csv', default=False, type=bool)
        dataset_type = request.args.get('dataset_type', type=str)

        if download_requested and dataset_type:
            return serve_preprocessed_file(dataset_id, dataset_type)

        cursor.execute("SELECT * FROM preprocessing_results WHERE dataset_id = %s", (dataset_id,))
        result = cursor.fetchone()

        if not result:
            return jsonify({"error": "No preprocessing results found for this dataset ID."}), 404

        return jsonify({
            "metadata": {
                "X_train_shape": result["X_train_shape"],
                "X_val_shape": result["X_val_shape"],
                "X_test_shape": result["X_test_shape"],
                "y_train_shape": result["y_train_shape"],
                "y_val_shape": result["y_val_shape"],
                "y_test_shape": result["y_test_shape"]
            },
            "logs": result["logs"].split("\n") if result["logs"] else [],
            "download_endpoints": {
                k: f"/preprocessing_results/{dataset_id}?download_csv=true&dataset_type={k}"
                for k in ["train_features", "train_labels",
                         "val_features", "val_labels",
                         "test_features", "test_labels"]
            }
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

@app.route('/rl_step', methods=['POST'])
def rl_step():
    """Take a reinforcement learning step (placeholder)."""
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    action = data.get('action')

    if not dataset_id or action is None:
        return jsonify({"error": "Dataset ID and action are required."}), 400

    try:
        # This would use a global env variable if implemented
        return jsonify({"error": "Environment not initialized."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feature_lookup', methods=['GET'])
def get_feature_lookup():
    """
    Return encoded feature names (post-preprocessing) in the exact order used for model training.
    """
    dataset_id = request.query_string.decode() and request.args.get('dataset_id')
    if dataset_id in (None, "", "null", "undefined"):
        return jsonify({"status": "error", "message": "dataset_id parameter is required"}), 400
    try:
        dataset_id = int(dataset_id)
    except ValueError:
        return jsonify({"status": "error", "message": "dataset_id must be an integer"}), 400

    preproc_path = os.path.abspath(f"preprocessor_pipeline_{dataset_id}.joblib")
    if not os.path.exists(preproc_path):
        return jsonify({
            "status": "error",
            "message": (
                f"Preprocessor not found for dataset_id={dataset_id}. "
                f"Run /preprocess first to fit & save the pipeline."
            )
        }), 404

    def _cols_to_list(cols):
        if cols is None:
            return []
        if isinstance(cols, slice):
            start = 0 if cols.start is None else cols.start
            stop = cols.stop
            step = 1 if cols.step is None else cols.step
            if isinstance(start, int) and isinstance(stop, int):
                return list(range(start, stop, step))
            return [f"slice_{start}_{stop}_{step}"]
        if isinstance(cols, (list, tuple)):
            return list(cols)
        if isinstance(cols, (np.ndarray, pd.Index)):
            return list(cols.tolist())
        return [cols]

    def _pretty(n: str) -> str:
        for prefix in ("all__", "num__", "cat__"):
            if isinstance(n, str) and n.startswith(prefix):
                return n[len(prefix):]
        return n

    try:
        preproc = joblib_load(preproc_path)

        names = []
        raw_to_encoded = {}

        try:
            out_names = preproc.get_feature_names_out()
            if out_names is not None and len(out_names) > 0:
                pretty_names = [_pretty(n) for n in out_names]
                for i, pname in enumerate(pretty_names):
                    raw_to_encoded.setdefault(str(pname), []).append(i)
                feature_map = {i: pretty_names[i] for i in range(len(pretty_names))}
                return jsonify({
                    "status": "success",
                    "n_features": len(pretty_names),
                    "feature_map": feature_map,
                    "raw_to_encoded": raw_to_encoded
                })
        except Exception:
            pass

        for step_name, transformer, cols in getattr(preproc, "transformers_", []):
            col_list = _cols_to_list(cols)
            if not col_list:
                continue

            if step_name == "all":
                start = len(names)
                for i, c in enumerate(col_list):
                    cname = str(c)
                    names.append(f"all__{cname}")
                    raw_to_encoded.setdefault(cname, []).append(start + i)
                continue

            if step_name == "num":
                start = len(names)
                for i, c in enumerate(col_list):
                    cname = str(c)
                    names.append(f"num__{cname}")
                    raw_to_encoded.setdefault(cname, []).append(start + i)
                continue

            if step_name == "cat":
                pipe = transformer
                ohe = getattr(pipe, "named_steps", {}).get("onehot", None)
                if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                    cat_feature_names = ohe.get_feature_names_out(col_list)
                    start = len(names)
                    for i, full in enumerate(cat_feature_names):
                        names.append(f"cat__{full}")
                    idx = start
                    for col, cats in zip(col_list, getattr(ohe, "categories_", [])):
                        count = len(cats)
                        raw_to_encoded.setdefault(str(col), []).extend(range(idx, idx + count))
                        idx += count
                else:
                    start = len(names)
                    for i, c in enumerate(col_list):
                        cname = str(c)
                        names.append(f"cat__{cname}")
                        raw_to_encoded.setdefault(cname, []).append(start + i)
                continue

            if step_name == "ip":
                start = len(names)
                for i, c in enumerate(col_list):
                    cname = str(c)
                    octets = [f"{cname}_o1", f"{cname}_o2", f"{cname}_o3", f"{cname}_o4"]
                    names.extend([f"ip__{o}" for o in octets])
                    base = start + i * 4
                    raw_to_encoded.setdefault(cname, []).extend([base, base + 1, base + 2, base + 3])
                continue

        pretty_names = [_pretty(n) for n in names]
        feature_map = {i: pretty_names[i] for i in range(len(pretty_names))}
        return jsonify({
            "status": "success",
            "n_features": len(pretty_names),
            "feature_map": feature_map,
            "raw_to_encoded": raw_to_encoded
        })

    except Exception as e:
        logger.error("Error in /feature_lookup", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Could not build feature lookup: {e}"
        }), 500

@app.route('/train_policy_model', methods=['GET'])
def train_policy_model_route():
   
    global TRAINING_IN_PROGRESS
    if TRAINING_IN_PROGRESS:
        logger.warning("Attempted to start a new training run while one was already in progress.")
        error_payload = {
            'error': 'A training process is already running. Please wait for it to finish or restart the server.',
            'stage': 'error_server_busy'
        }
        def error_stream():
            yield f"data: {json.dumps(error_payload)}\n\n"
        return Response(stream_with_context(error_stream()), mimetype="text/event-stream")

    TRAINING_IN_PROGRESS = True

    try:
        classifier_type_arg = request.args.get("classifier", "rf").lower()
        if classifier_type_arg not in ["dt", "logreg", "knn", "rf", "svm"]:
            classifier_type_arg = "rf"

        dataset_id = int(request.args.get("dataset_id"))

        policy_config_overrides = {
            'classifier_spec': {'name': classifier_type_arg, 'params': {}},
            'max_epochs': int(request.args.get("epochs", 200)),
            'warmup_steps': int(request.args.get("warmup_steps", 64)),
            'sac_batch_size': int(request.args.get("batch_size", 32)),
            'sac_buffer_size': int(request.args.get("buffer_size", 10000)),
            'learning_rate': float(request.args.get("learning_rate", 3e-5)),
            'actor_learning_rate': float(request.args.get("actor_learning_rate", 3e-5)),
            'alpha_learning_rate': float(request.args.get("alpha_learning_rate", 3e-4)),
            'gamma': float(request.args.get("gamma", 0.99)),
            'tau': float(request.args.get("tau", 0.001)),
            'target_entropy_ratio': float(request.args.get("target_entropy_ratio", 0.98)),
        }

    except (ValueError, TypeError) as e:
        logger.error(f"Parameter error in /train_policy_model: {str(e)}", exc_info=True)
        TRAINING_IN_PROGRESS = False
        return jsonify({"error": f"Invalid parameter: {e}", "stage": "policy_param_error"}), 400

    def generate_policy_training_stream():
        global TRAINING_IN_PROGRESS
        try:
            X_train_np, y_train_np, X_val_np, y_val_np = load_training_data(dataset_id)
            if X_train_np is None or y_train_np is None:
                error_msg = f"Failed to load real training data for dataset_id: {dataset_id}."
                yield f"data: {json.dumps({'error': error_msg, 'stage': 'policy_data_load_error'})}\n\n"
                return

            X_train_ready = X_train_np
            X_val_ready = X_val_np

            max_features = X_train_ready.shape[1]
            policy_learner = PolicyLearner(max_features=max_features, config=policy_config_overrides)
            logger.info(f"PolicyLearner initialized with {classifier_type_arg} classifier.")

            last_selected_features = []
            api_epoch_num = 0
            for api_epoch_num in range(1, policy_config_overrides['max_epochs'] + 1):
                updates_generator = policy_learner.run_training_step(
                    real_data_tuple=(X_train_ready, y_train_np),
                    val_data=(X_val_ready, y_val_np),
                    epoch=api_epoch_num
                )
                for update in updates_generator:
                    if isinstance(update, dict) and 'selected_features' in update:
                        sf = update.get('selected_features') or []
                        if isinstance(sf, (list, tuple)):
                            last_selected_features = list(sf)
                    yield f"data: {json.dumps(tensor_to_serializable(update))}\n\n"

            best_f1 = float(policy_learner.best_f1_so_far)
            best_features_sorted = sorted(list(policy_learner.best_features_so_far))

            if not last_selected_features:
                last_selected_features = best_features_sorted
            final_features_sorted = sorted(set(int(f) for f in last_selected_features))

            final_log = {
                'stage': 'policy_training_complete',
                'message': f'Training complete after {api_epoch_num} epochs.',
                'classifier_used': classifier_type_arg,
                'final_features': final_features_sorted,
                'best_f1_score': best_f1,
                'best_features': best_features_sorted,
            }
            logger.info(f"Policy training complete. Final (testing) set: {final_features_sorted}")
            yield f"data: {json.dumps(policy_learner._make_json_serializable(final_log))}\n\n"

        except Exception as e:
            logger.error(f"Fatal error in policy training stream: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'type': type(e).__name__, 'stage': 'policy_error_fatal'})}\n\n"
        finally:
            TRAINING_IN_PROGRESS = False
            logger.info("Training process finished. Server is now available.")

    return Response(
        stream_with_context(generate_policy_training_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

@app.route('/test_policy', methods=['POST'])
def test_policy_route():
 
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        feature_set = data.get('features')
        classifier_spec_dict = data.get('classifier_spec')

        if not isinstance(feature_set, list) or dataset_id is None or not isinstance(classifier_spec_dict, dict):
            return jsonify({'error': 'Missing or invalid dataset_id, features list, or classifier_spec.'}), 400

        classifier_spec = ClassifierSpec(**classifier_spec_dict)
        classifier_name = classifier_spec.name.upper()

        supported_names = ["dt", "logreg", "knn", "rf", "svm"]
        if classifier_spec.name not in supported_names:
            return jsonify({'error': f"Unsupported classifier: {classifier_spec.name}. Supported: {supported_names}"}), 400

        X_train, y_train = load_data_split(dataset_id, 'train')
        X_test, y_test = load_data_split(dataset_id, 'test')

        if X_train is None or X_test is None:
            return jsonify({'error': f'Could not load data for dataset {dataset_id}.'}), 404

        policy_feature_mask = np.zeros(X_train.shape[1], dtype=bool)
        valid_features = [f for f in feature_set if isinstance(f, int) and 0 <= f < X_train.shape[1]]
        if not valid_features:
            return jsonify({'error': 'The provided feature set is empty or invalid.'}), 400
        policy_feature_mask[valid_features] = True

        policy_model = build_classifier(classifier_spec)

        policy_train_start = time.time()
        policy_model.fit(X_train[:, policy_feature_mask], y_train)
        policy_training_time = time.time() - policy_train_start

        policy_infer_start = time.time()
        y_pred_policy = policy_model.predict(X_test[:, policy_feature_mask])
        policy_inference_time = time.time() - policy_infer_start

        policy_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred_policy)),
            'f1_score': float(f1_score(y_test, y_pred_policy, average='weighted', zero_division=0)),
            'precision': float(precision_score(y_test, y_pred_policy, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_policy, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred_policy).tolist(),
            'features_tested': valid_features,
            'classifier_used': classifier_name,
            'training_time': round(policy_training_time, 4),
            'inference_time': round(policy_inference_time, 4)
        }

        all_features_model = build_classifier(classifier_spec)

        all_features_train_start = time.time()
        all_features_model.fit(X_train, y_train)
        all_features_training_time = time.time() - all_features_train_start

        all_features_infer_start = time.time()
        y_pred_all_features = all_features_model.predict(X_test)
        all_features_inference_time = time.time() - all_features_infer_start

        all_features_metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred_all_features)),
            'f1_score': float(f1_score(y_test, y_pred_all_features, average='weighted', zero_division=0)),
            'features_tested': 'all',
            'classifier_used': classifier_name,
            'training_time': round(all_features_training_time, 4),
            'inference_time': round(all_features_inference_time, 4)
        }

        results = {
            'message': 'Testing complete. Models are ready for download.',
            'policy_performance': policy_metrics,
            'all_features_performance': all_features_metrics,
            'num_test_samples': int(X_test.shape[0]),
            'download_available': True,
            'export_endpoint': '/export_model',
            'export_payload': {
                'dataset_id': dataset_id,
                'features': valid_features,
                'classifier_spec': classifier_spec_dict,
                'export_baseline': True
            }
        }
        return jsonify(results)

    except Exception as e:
        logger.error(f"Error during policy testing: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route("/export_model", methods=["POST"])
def export_model():
    """
    Train portable pipelines for Docker inference service.
    """
    try:
        data = request.get_json(force=True)
        dataset_id = data.get("dataset_id")
        feature_set = data.get("features")
        classifier_spec_dict = data.get("classifier_spec")
        use_scaler = bool(data.get("use_scaler", False))
        export_baseline = bool(data.get("export_baseline", True))

        if dataset_id is None or not isinstance(feature_set, list) or not isinstance(classifier_spec_dict, dict):
            return jsonify({"error": "Missing/invalid dataset_id, features, or classifier_spec"}), 400

        classifier_spec = ClassifierSpec(**classifier_spec_dict)
        supported = ["dt", "logreg", "knn", "rf", "svm"]
        if classifier_spec.name not in supported:
            return jsonify({"error": f"Unsupported classifier: {classifier_spec.name}. Supported: {supported}"}), 400

        X_train, y_train = load_data_split(dataset_id, "train")
        X_val, y_val = load_data_split(dataset_id, "val")
        X_test, y_test = load_data_split(dataset_id, "test")

        if X_train is None or X_val is None:
            return jsonify({"error": f"Could not load train/val for dataset {dataset_id}"}), 404

        X_tv = np.vstack([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])

        d = int(X_tv.shape[1])
        indices = [int(i) for i in feature_set if isinstance(i, (int, np.integer)) and 0 <= i < d]
        
        ts = time.strftime("%Y%m%d-%H%M%S")
        cls_name = classifier_spec.name.upper()
        
        policy_meta = None
        baseline_meta = None

        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            
            if indices:
                policy_pipeline = build_inference_pipeline(classifier_spec, indices, use_scaler=use_scaler)

                t0 = time.time()
                policy_pipeline.fit(X_tv, y_tv)
                policy_train_time = round(time.time() - t0, 4)

                policy_infer_time = None
                if X_test is not None and X_test.shape[0] > 0:
                    t1 = time.time()
                    _ = policy_pipeline.predict(X_test)
                    policy_infer_time = round(time.time() - t1, 4)

                policy_meta = {
                    "exported_at": ts,
                    "model_type": "policy_selected_features",
                    "classifier": cls_name,
                    "feature_indices": indices,
                    "use_scaler": use_scaler,
                    "train_plus_val_samples": int(X_tv.shape[0]),
                    "n_features_total": d,
                    "train_time_sec": policy_train_time,
                    "test_infer_time_sec": policy_infer_time,
                    "expected_input_dim": len(indices)
                }

                zf.writestr("policy/metadata.json", json.dumps(policy_meta, indent=2))
                
                with io.BytesIO() as model_buffer:
                    dump(policy_pipeline, model_buffer)
                    model_buffer.seek(0)
                    zf.writestr("policy/model.joblib", model_buffer.read())
            else:
                logger.warning("No valid features provided for policy model. Skipping policy export.")

            if export_baseline:
                all_feature_indices = list(range(d))
                baseline_pipeline = build_inference_pipeline(classifier_spec, all_feature_indices, use_scaler=use_scaler)

                b0 = time.time()
                baseline_pipeline.fit(X_tv, y_tv)
                baseline_train_time = round(time.time() - b0, 4)

                baseline_infer_time = None
                if X_test is not None and X_test.shape[0] > 0:
                    b1 = time.time()
                    _ = baseline_pipeline.predict(X_test)
                    baseline_infer_time = round(time.time() - b1, 4)

                baseline_meta = {
                    "exported_at": ts,
                    "model_type": "baseline_all_features",
                    "classifier": cls_name,
                    "feature_indices": all_feature_indices,
                    "use_scaler": use_scaler,
                    "train_plus_val_samples": int(X_tv.shape[0]),
                    "n_features_total": d,
                    "train_time_sec": baseline_train_time,
                    "test_infer_time_sec": baseline_infer_time,
                    "expected_input_dim": d
                }
                
                zf.writestr("baseline/metadata.json", json.dumps(baseline_meta, indent=2))
                
                with io.BytesIO() as model_buffer:
                    dump(baseline_pipeline, model_buffer)
                    model_buffer.seek(0)
                    zf.writestr("baseline/model.joblib", model_buffer.read())

        mem_zip.seek(0)
        zip_name = f"irdf_models_{classifier_spec.name}_{ts}.zip"
        return send_file(mem_zip, mimetype="application/zip", as_attachment=True, download_name=zip_name)

    except Exception as e:
        logger.error(f"Error during model export: {e}", exc_info=True)
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
