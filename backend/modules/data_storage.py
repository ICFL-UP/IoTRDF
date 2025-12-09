"""
Data Storage Module
Handles persistence of preprocessed dataset splits to CSV files and database
Used by the main API to save train/val/test splits for model training
"""
import numpy as np
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_OUTPUT_DIR = Path("./preprocessed_datasets")

def save_preprocessed_splits(cursor, dataset_id: str, split_name: str, X: np.ndarray, y: np.ndarray):
    
    try:
        DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        x_path = DATASET_OUTPUT_DIR / f"{dataset_id}_{split_name}_X.csv"
        y_path = DATASET_OUTPUT_DIR / f"{dataset_id}_{split_name}_y.csv"
        np.savetxt(x_path, X, delimiter=',')
        np.savetxt(y_path, y, delimiter=',', fmt='%d')
        logger.info(f"Saved {split_name} splits to CSV at {DATASET_OUTPUT_DIR}")

        table_map = {
            "train": "training_data",
            "val": "validation_data",
            "test": "testing_data"
        }
        table_name = table_map[split_name]
        
        X_db = json.dumps(X.tolist())
        y_db = json.dumps(y.tolist())

  
        cursor.execute(f"DELETE FROM {table_name} WHERE dataset_id = %s", (dataset_id,))
        cursor.execute(
            f"INSERT INTO {table_name} (dataset_id, X_{split_name}, y_{split_name}) VALUES (%s, %s, %s)",
            (dataset_id, X_db, y_db)
        )
        logger.info(f"Saved {split_name} splits to database table '{table_name}'.")

    except Exception as e:
        logger.error(f"Failed to save {split_name} data: {e}")
       
        raise