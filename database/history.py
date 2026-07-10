"""
Database History Module
SQLite persistence for experiment history and model versions.
"""

import sqlite3
import os
import json
import pickle
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "optiml_history.db")


def init_database(db_path=None):
    """
    Initialize the SQLite database and create tables if they don't exist.
    
    Returns:
        sqlite3.Connection
    """
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            name TEXT,
            n_factors INTEGER,
            n_responses INTEGER,
            n_experiments INTEGER,
            factor_names TEXT,
            response_names TEXT,
            best_model TEXT,
            best_r2 REAL,
            best_value REAL,
            n_iterations INTEGER,
            metadata TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER,
            model_name TEXT,
            r2 REAL,
            cv_r2 REAL,
            rmse REAL,
            params TEXT,
            created_at TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
    """)

    conn.commit()
    return conn


def save_experiment(conn, data: dict) -> int:
    """
    Save an experiment session to the database.
    
    Args:
        conn: SQLite connection
        data: Session data dict
        
    Returns:
        int: Experiment ID
    """
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO experiments (timestamp, name, n_factors, n_responses, n_experiments,
                                  factor_names, response_names, best_model, best_r2,
                                  best_value, n_iterations, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        data.get("name", "Unnamed Experiment"),
        len(data.get("factor_cols", [])),
        len(data.get("response_cols", [])),
        data.get("n_experiments", 0),
        json.dumps(data.get("factor_cols", [])),
        json.dumps(data.get("response_cols", [])),
        data.get("best_model_name", ""),
        data.get("best_r2", 0),
        data.get("best_value", 0),
        data.get("iteration_count", 0),
        json.dumps(data.get("metadata", {})),
    ))

    conn.commit()
    return cursor.lastrowid


def save_model_version(conn, experiment_id: int, model_name: str, metrics: dict, params: dict) -> int:
    """
    Save a model version to the database.
    
    Args:
        conn: SQLite connection
        experiment_id: Parent experiment ID
        model_name: Name of the model
        metrics: Performance metrics
        params: Hyperparameters
        
    Returns:
        int: Model version ID
    """
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_versions (experiment_id, model_name, r2, cv_r2, rmse, params, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        experiment_id,
        model_name,
        metrics.get("R²", 0),
        metrics.get("CV_R²_mean", 0),
        metrics.get("RMSE", 0),
        json.dumps(params),
        datetime.now().isoformat(),
    ))

    conn.commit()
    return cursor.lastrowid


def load_experiment_history(conn):
    """
    Load all past experiments.
    
    Returns:
        pd.DataFrame or None
    """
    import pandas as pd

    try:
        df = pd.read_sql_query("""
            SELECT id, timestamp, name, n_factors, n_responses, n_experiments,
                   best_model, best_r2, n_iterations
            FROM experiments
            ORDER BY timestamp DESC
        """, conn)
        return df
    except Exception:
        return None


def load_model_versions(conn, experiment_id: int):
    """
    Load model versions for a specific experiment.
    
    Returns:
        pd.DataFrame or None
    """
    import pandas as pd

    try:
        df = pd.read_sql_query("""
            SELECT id, model_name, r2, cv_r2, rmse, created_at
            FROM model_versions
            WHERE experiment_id = ?
            ORDER BY cv_r2 DESC
        """, conn, params=(experiment_id,))
        return df
    except Exception:
        return None
