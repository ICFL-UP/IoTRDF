"""
data_splitter.py

Provides the DataSplitter utility used in the IRDF backend to perform
stratified train/validation/test splits and optionally apply class
imbalance handling via downsampling or class weighting. The resulting
splits can be persisted to the database for reuse during training.
"""
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample  
import numpy as np
import pandas as pd  
import json
import logging

logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, cleaned_data, dataset_id, cursor):
        """
        Initializes the DataSplitter with cleaned data and a database cursor.
        """
        self.cleaned_data = cleaned_data
        self.dataset_id = dataset_id
        self.cursor = cursor

    def _can_stratify(self, y):
        """Checks if each class has at least 2 samples for stratified splitting."""
        _, counts = np.unique(y, return_counts=True)
        return np.all(counts >= 2)

    def _maybe_compute_class_weights(self, y_train, imbalance_ratio_threshold):
        """
        Computes class weights if the class imbalance ratio exceeds a threshold.
        """
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) <= 1:
            return None
        min_count = counts.min()
        if min_count == 0:
            return None
        ratio = counts.max() / min_count
        if ratio >= imbalance_ratio_threshold:
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            cw = dict(zip(map(int, classes), map(float, weights)))
            logger.info(f"Class imbalance detected (ratio={ratio:.2f}). Applying class weights: {cw}")
            return cw
        else:
            logger.info(f"Class distribution is balanced (ratio={ratio:.2f}). No class weights applied.")
            return None

    
    def _downsample(self, X_train, y_train, random_state):
        """
        Downsamples the majority classes in the training set to match the smallest class.
        """
        logger.info("Applying downsampling to balance the training data...")
        
        df_train = pd.DataFrame(X_train)
        df_train['label'] = y_train
        
        min_class_size = df_train['label'].value_counts().min()
        if min_class_size < 1:
            logger.warning("Smallest class has 0 samples. Skipping downsampling.")
            return X_train, y_train

        logger.info(f"Smallest class has {min_class_size} samples. Downsampling others to match.")

        df_balanced = pd.concat([
            resample(group,
                     replace=False,
                     n_samples=min_class_size,
                     random_state=random_state)
            for _, group in df_train.groupby('label')
        ]).reset_index(drop=True)

        y_train_balanced = df_balanced['label'].values
        X_train_balanced = df_balanced.drop(columns=['label']).values

        logger.info(f"Downsampling complete. New training set shape: {X_train_balanced.shape}")
        
        return X_train_balanced, y_train_balanced
 

    def _save_to_database(self, table_name, X_data, y_data):
        """
        Saves a data split to the specified database table.
        """
        suffix_map = {
            "training_data": ("X_train", "y_train"),
            "validation_data": ("X_val", "y_val"),
            "testing_data": ("X_test", "y_test"),
        }
        if table_name not in suffix_map:
            raise ValueError(f"Unknown table name: {table_name}")

        x_col, y_col = suffix_map[table_name]
        X_json = json.dumps(np.asarray(X_data).tolist())
        y_json = json.dumps(np.asarray(y_data).tolist())

        self.cursor.execute(f"DELETE FROM {table_name} WHERE dataset_id = %s", (self.dataset_id,))
        self.cursor.execute(
            f"INSERT INTO {table_name} (dataset_id, {x_col}, {y_col}) VALUES (%s, %s, %s)",
            (self.dataset_id, X_json, y_json),
        )
        logger.info(f"Data successfully saved to `{table_name}`.")

    @staticmethod
    def _class_dist(y):
        """Return class distribution as counts and ratios."""
        classes, counts = np.unique(y, return_counts=True)
        total = counts.sum() if counts.size else 1
        ratios = counts / total
        return (
            {int(c): int(n) for c, n in zip(classes, counts)},
            {int(c): float(r) for c, r in zip(classes, ratios)},
        )

    def split_dataset(
        self,
        test_size=0.20,
        val_size=0.10,
        random_state=42,
        imbalance_ratio_threshold=1.5,
        stratify_val=False,
        persist=True,
        apply_downsampling=True,  # <
    ):
        """
        Splits data and optionally applies downsampling to the training set.
        """
        try:
            X = np.asarray(self.cleaned_data["X"])
            y = np.asarray(self.cleaned_data["y"])

            if y.ndim > 1:
                if y.shape[1] == 1:
                    y = y.ravel()
                else:
                    raise ValueError("Invalid y shape: expected 1D class labels.")

            # Step 1: Stratified TEST split
            strat_all = y if self._can_stratify(y) else None
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=strat_all
            )

            # Step 2: TRAIN/VAL split
            remaining = 1.0 - test_size
            rel_val = val_size / remaining
            strat_tv = y_trainval if (stratify_val and self._can_stratify(y_trainval)) else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=rel_val, random_state=random_state, stratify=strat_tv
            )

            class_weights = None
            # --- UPDATED LOGIC TO INCLUDE DOWNSAMPLING ---
            if apply_downsampling:
                # If we downsample, the training data becomes balanced and class weights are not needed.
                X_train, y_train = self._downsample(X_train, y_train, random_state)
            else:
                # If not, we compute class weights for the imbalanced data as before.
                class_weights = self._maybe_compute_class_weights(y_train, imbalance_ratio_threshold)
            # --- END OF UPDATED LOGIC ---

            if persist:
                self._save_to_database("training_data", X_train, y_train)
                self._save_to_database("validation_data", X_val, y_val)
                self._save_to_database("testing_data", X_test, y_test)

            train_cnts, train_rat = self._class_dist(y_train)
            val_cnts, val_rat = self._class_dist(y_val)
            test_cnts, test_rat = self._class_dist(y_test)

            logger.info("Dataset successfully split and %s.",
                        "saved to the database" if persist else "kept in memory")
            logger.info(f"Final Shapes -> Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
            logger.info(f"Class dist (counts) -> Train: {train_cnts}, Val: {val_cnts}, Test: {test_cnts}")
            logger.info(f"Class dist (ratios) -> Train: {train_rat}, Val: {val_rat}, Test: {test_rat}")

            return {
                "X_train": X_train, "y_train": y_train,
                "X_val": X_val, "y_val": y_val,
                "X_test": X_test, "y_test": y_test,
                "class_weights": class_weights,
                "logs": [
                    "Dataset split successfully.",
                    f"Train Shape: {X_train.shape}",
                    f"Validation Shape: {X_val.shape}",
                    f"Test Shape: {X_test.shape}",
                    f"Train class counts: {train_cnts}",
                ],
            }

        except Exception as e:
            logger.error(f"Error splitting the dataset: {e}")
            raise