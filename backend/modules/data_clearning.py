"""
data_cleaning.py

Implements the DataCleaner class used by the IRDF backend to load the selected
dataset from MySQL, perform cleaning and preprocessing, and return a
model-ready feature matrix X and label vector y. The pipeline handles label
filtering, timestamp expansion, IP address parsing, numeric scaling, and
categorical one-hot encoding, and saves the fitted preprocessing pipeline as
a Joblib artifact for reuse.
"""
import os
import logging
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

logger = logging.getLogger(__name__)


OHE_MAX_CARDINALITY = 100


def to_str_block(X_df: pd.DataFrame) -> pd.DataFrame:
    """Safely cast any block to string dtype for OHE."""
    return pd.DataFrame(X_df).astype(str)

def parse_ip_addresses(X_df: pd.DataFrame) -> np.ndarray:
    
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)

    all_octets = []
    for col in X_df.columns:
        octets = X_df[col].astype(str).str.split('.', expand=True, n=3)
        for i in range(4):
            if i not in octets.columns:
                octets[i] = np.nan
        octets.columns = [f'{col}_o1', f'{col}_o2', f'{col}_o3', f'{col}_o4']
        for oc in octets.columns:
            octets[oc] = pd.to_numeric(octets[oc], errors='coerce')
        octets = octets.fillna(0)
        all_octets.append(octets)

    return pd.concat(all_octets, axis=1).to_numpy(dtype=np.float32)

class DataCleaner:
    def __init__(
        self,
        connection,
        dataset_id: int,
        random_state: int = 42,
        allowed_labels: Optional[List[str]] = None,
        label_col: str = "Attack_type",
    ):
        self.connection = connection
        self.dataset_id = dataset_id
        self.random_state = random_state
        self.label_col = label_col
        self.allowed_labels = allowed_labels or ["Normal", "Ransomware"]
        self.preprocessor_path = f"preprocessor_pipeline_{dataset_id}.joblib"

    def load_and_clean(self) -> Dict:
        
        cursor = None
        try:
            logger.info(f"Starting data preparation for dataset_id={self.dataset_id}.")
            cursor = self.connection.cursor(dictionary=True)

            # Resolve dataset path
            logger.info("Fetching dataset path from the database...")
            cursor.execute("SELECT file_path FROM datasets WHERE id = %s;", (self.dataset_id,))
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"No dataset found with ID {self.dataset_id}")

            file_path = os.path.abspath(row["file_path"])
            logger.info(f"Dataset file path resolved: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found at: {file_path}")

            # Load CSV
            logger.info(f"Loading dataset from: {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            if df.empty:
                raise ValueError("Dataset is empty.")
            logger.info(f"Dataset loaded. Initial shape: {df.shape}")

            # Clean + preprocess
            cleaned = self.clean_and_preprocess(df)
            logger.info("Data cleaning and preprocessing completed successfully.")
            return cleaned

        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            if cursor:
                cursor.close()

    def clean_and_preprocess(self, df: pd.DataFrame) -> Dict:
        
        logger.info("--- Starting ADVANCED data preprocessing pipeline (for automatic selection) ---")
        timestamp_col = None

        # 1) Filter to allowed labels
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found.")
        df = df.dropna(subset=[self.label_col]).copy()
        df[self.label_col] = df[self.label_col].astype(str).str.strip()
        keep_labs = {s.strip().lower() for s in self.allowed_labels}
        before_rows = len(df)
        df = df[df[self.label_col].str.lower().isin(keep_labs)].copy()
        logger.info(f"EARLY FILTER: kept only {self.allowed_labels}. Rows {before_rows} -> {len(df)}.")
        if df.empty:
            raise ValueError(f"No rows remain after filtering to {self.allowed_labels}.")
        logger.info(f"Post-filter unique labels: {sorted(df[self.label_col].unique().tolist())}")

        # 2) Drop all-NaN columns, drop duplicates
        all_nan_cols = [c for c in df.columns if df[c].isna().all()]
        if all_nan_cols:
            df = df.drop(columns=all_nan_cols)
            logger.info(f"Dropped all-NaN columns: {all_nan_cols}")
        before_dups = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped_dups = before_dups - len(df)
        if dropped_dups:
            logger.info(f"Dropped {dropped_dups} duplicate rows.")

        # 3) Timestamp features
        timestamp_col = next((c for c in df.columns if "time" in c.lower()), None)
        if timestamp_col:
            logger.info(f"Processing timestamp column: '{timestamp_col}'")
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            keep_mask = df[timestamp_col].notna()
            dropped_bad = int((~keep_mask).sum())
            df = df.loc[keep_mask].copy()
            df["year"] = df[timestamp_col].dt.year
            df["month"] = df[timestamp_col].dt.month
            df["day"] = df[timestamp_col].dt.day
            df["hour"] = df[timestamp_col].dt.hour
            df = df.drop(columns=[timestamp_col])
            logger.info(
                "Timestamp features created."
                + (f" Dropped {dropped_bad} rows with invalid dates." if dropped_bad else "")
            )

        # 4) Split X/y, drop classic leakage
        logger.info("Separating features (X) from labels (y).")
        y_series = df[self.label_col]
        X = df.drop(columns=[self.label_col])

        leak_cols = [c for c in X.columns if c.lower() == "attack_label"]
        if leak_cols:
            X = X.drop(columns=leak_cols)
            logger.info(f"Dropped potential leakage column(s): {leak_cols}")

        # --- NEW: LOG raw X features & y column RIGHT HERE ---
        logger.info(f"Raw X (pre-typing/encoding) column list ({X.shape[1]} columns): {list(X.columns)}")
        logger.info(f"Y column name: '{self.label_col}'")

        # 5) Detect types
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # IP-like categoricals (parse to 4 octets instead of OHE)
        ip_features = [c for c in categorical_features if 'ip' in c.lower() and ('addr' in c.lower() or 'host' in c.lower())]
        if ip_features:
            categorical_features = [c for c in categorical_features if c not in ip_features]

        # Limit OHE to low-cardinality categoricals
        cat_cardinalities = {c: int(X[c].nunique(dropna=True)) for c in categorical_features}
        low_card_cat_features = [c for c in categorical_features if cat_cardinalities[c] <= OHE_MAX_CARDINALITY]
        high_card_cat_features = [c for c in categorical_features if c not in low_card_cat_features]

        logger.info(f"Detected numeric cols: {len(numeric_features)} -> {numeric_features}")
        logger.info(f"Detected IP-like cols: {len(ip_features)} -> {ip_features}")
        logger.info(f"Detected categorical cols: {len(categorical_features)} "
                    f"(low-card for OHE: {len(low_card_cat_features)} -> {low_card_cat_features}, "
                    f"high-card skipped: {len(high_card_cat_features)} -> {high_card_cat_features})")

        # 6) Build pipelines
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        ip_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="0.0.0.0")),
            ("parser", FunctionTransformer(parse_ip_addresses)),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("caster", FunctionTransformer(to_str_block)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("ip",  ip_pipeline,      ip_features),
                ("cat", categorical_pipeline, low_card_cat_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        logger.info(
            "Encoding plan: "
            f"NUM (impute+scale)={len(numeric_features)}, "
            f"IP (parse->4 octets + scale)={len(ip_features)}, "
            f"CAT (OHE)={len(low_card_cat_features)}"
        )

        # 7) Fit ColumnTransformer on ALL data (leakage by request) and save
        logger.warning("Fitting preprocessor on the entire dataset. This introduces data leakage (as requested).")
        X_processed = preprocessor.fit_transform(X)
        dump(preprocessor, self.preprocessor_path)
        logger.info(f"Preprocessor saved to: {self.preprocessor_path}. Encoded X shape: {X_processed.shape}")

        # 8) Encode labels (fixed order), drop any invalid rows, then shuffle
        label_mapping = {lab: i for i, lab in enumerate(self.allowed_labels)}
        lower_map = {lab.lower(): idx for lab, idx in label_mapping.items()}
        y_mapped = y_series.astype(str).str.strip().str.lower().map(lower_map)

        valid_mask = y_mapped.notna()
        if not valid_mask.all():
            dropped = int((~valid_mask).sum())
            logger.warning(f"Dropping {dropped} rows with labels outside {self.allowed_labels}.")
            y_mapped = y_mapped[valid_mask]
            X_processed = X_processed[valid_mask.to_numpy()]

        y_final = y_mapped.to_numpy(dtype=np.int64)

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(y_final))
        X_final, y_final = X_processed[idx], y_final[idx]
        logger.info(f"Data shuffled using random state {self.random_state}.")

        # 9) Build final encoded feature names (in order) and LOG them
        feature_names: List[str] = []
        logger.info("Constructing final encoded feature names in-order ...")
        for name, pipe, original_cols in preprocessor.transformers_:
            if not original_cols:
                continue
            if name == "num":
                feature_names.extend(list(original_cols))
            elif name == "ip":
                for col in list(original_cols):
                    feature_names.extend([f"{col}_o1", f"{col}_o2", f"{col}_o3", f"{col}_o4"])
            elif name == "cat":
                try:
                    ohe = pipe.named_steps["onehot"]
                    feature_names.extend(ohe.get_feature_names_out(list(original_cols)))
                except Exception as err:
                    logger.warning(f"Could not get OHE feature names, falling back. Details: {err}")
                    feature_names.extend([f"{c}__enc" for c in list(original_cols)])

        
        logger.info(f"Final feature column list (X, {len(feature_names)} columns): {feature_names}")
        logger.info(f"Final X shape: {X_final.shape}; Final y length: {len(y_final)}")
        logger.info(f"Label column (y)='{self.label_col}', mapping={label_mapping}")

        classes, counts = np.unique(y_final, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(classes, counts)}
        logger.info(f"Class counts after cleaning/shuffle: {class_counts}")

        return {
            "X": X_final,
            "y": y_final,
            "metadata": {
                "num_samples": int(X_final.shape[0]),
                "num_features": int(X_final.shape[1]),
                "label_mapping": label_mapping,
                "feature_names": feature_names,
                "timestamp_used": bool(timestamp_col),
                "scaled_here": True,  
                "class_counts": class_counts,
            },
        }