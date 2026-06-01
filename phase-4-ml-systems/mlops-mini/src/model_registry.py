"""
model_registry.py — Track model versions with stage transitions.

Stages follow the MLflow Model Registry convention:
  development → staging → production → archived

Only one model per (name) can be in 'production' at a time.
Promoting a new version to 'production' automatically archives the previous one.
"""

import sqlite3
import json
import shutil
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL_MODELS = """
    CREATE TABLE IF NOT EXISTS models (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        version     TEXT    NOT NULL,
        stage       TEXT    NOT NULL DEFAULT 'development',
        model_path  TEXT,
        metrics     TEXT    NOT NULL DEFAULT '{}',
        config      TEXT    NOT NULL DEFAULT '{}',
        registered_at REAL,
        updated_at  REAL,
        UNIQUE(name, version)
    )
"""


def _get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(DDL_MODELS)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Track model versions with metadata and lifecycle stage transitions.

    Parameters
    ----------
    db_path   : str — path to SQLite database (default: "mlops.db")
    model_dir : str — directory where model files are stored (default: "models/")

    Typical workflow
    ----------------
        registry = ModelRegistry()
        model_id = registry.register('my_model', 'v1.0', 'weights.npy',
                                     metrics={'val_acc': 0.92},
                                     config={'hidden': 128})
        registry.transition_stage('my_model', 'v1.0', 'staging')
        registry.transition_stage('my_model', 'v1.0', 'production')
        prod = registry.get_production('my_model')
    """

    STAGES = ['development', 'staging', 'production', 'archived']

    def __init__(self, db_path: str = "mlops.db", model_dir: str = "models/"):
        self.db_path   = db_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        # Ensure tables exist
        conn = _get_conn(db_path)
        conn.close()

    def register(self, name: str, version: str, model_path: str,
                 metrics: dict, config: dict) -> int:
        """
        Register a new model version.

        Copies the model file to `model_dir/name/version/` and stores
        metadata in the database.

        Parameters
        ----------
        name       : str  — model family name (e.g. 'my_mlp')
        version    : str  — version string (e.g. 'v1.0', 'v1.1')
        model_path : str  — path to the model file (weights, pickle, etc.)
        metrics    : dict — evaluation metrics (e.g. {'val_acc': 0.92})
        config     : dict — hyperparameters used to train this version

        Returns
        -------
        int — model ID in the database
        """
        # Copy model file to registry directory
        dest_dir  = os.path.join(self.model_dir, name, version)
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.exists(model_path):
            dest_path = os.path.join(dest_dir, os.path.basename(model_path))
            shutil.copy2(model_path, dest_path)
        else:
            # Path doesn't exist — store reference only
            dest_path = model_path

        now  = datetime.now().timestamp()
        conn = _get_conn(self.db_path)

        try:
            cur = conn.execute(
                """INSERT INTO models
                   (name, version, stage, model_path, metrics, config,
                    registered_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (name, version, 'development',
                 dest_path, json.dumps(metrics), json.dumps(config),
                 now, now)
            )
            model_id = cur.lastrowid
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            raise ValueError(f"Model '{name}' version '{version}' already registered.")

        conn.close()
        return model_id

    def get_model(self, name: str, version: str) -> dict:
        """
        Return metadata dict for a specific model version.

        Raises ValueError if the model/version is not found.
        """
        conn = _get_conn(self.db_path)
        row  = conn.execute(
            "SELECT * FROM models WHERE name=? AND version=?",
            (name, version)
        ).fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"Model '{name}' version '{version}' not found.")

        return self._row_to_dict(row)

    def transition_stage(self, name: str, version: str, stage: str):
        """
        Move a model version to a new lifecycle stage.

        Rules:
        - stage must be one of: development, staging, production, archived
        - When transitioning to 'production', any current production model
          for the same name is automatically archived.

        Parameters
        ----------
        name    : str — model family name
        version : str — version to transition
        stage   : str — target stage
        """
        if stage not in self.STAGES:
            raise ValueError(f"Unknown stage '{stage}'. "
                             f"Must be one of: {self.STAGES}")

        conn = _get_conn(self.db_path)

        # Verify model exists
        row = conn.execute(
            "SELECT id FROM models WHERE name=? AND version=?",
            (name, version)
        ).fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"Model '{name}' version '{version}' not found.")

        now = datetime.now().timestamp()

        if stage == 'production':
            # Archive any existing production model for this name
            conn.execute(
                """UPDATE models SET stage='archived', updated_at=?
                   WHERE name=? AND stage='production' AND version!=?""",
                (now, name, version)
            )

        conn.execute(
            "UPDATE models SET stage=?, updated_at=? WHERE name=? AND version=?",
            (stage, now, name, version)
        )
        conn.commit()
        conn.close()

    def get_production(self, name: str) -> dict:
        """
        Get the current production model for a given model name.

        Returns the metadata dict, or raises ValueError if none is in production.
        """
        conn = _get_conn(self.db_path)
        row  = conn.execute(
            "SELECT * FROM models WHERE name=? AND stage='production'",
            (name,)
        ).fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"No production model found for '{name}'.")

        return self._row_to_dict(row)

    def list_versions(self, name: str):
        """
        Print all versions of a model family with their stages and metrics.
        """
        conn = _get_conn(self.db_path)
        rows = conn.execute(
            "SELECT * FROM models WHERE name=? ORDER BY registered_at",
            (name,)
        ).fetchall()
        conn.close()

        if not rows:
            print(f"No models registered under name '{name}'.")
            return

        print(f"\n{'=' * 70}")
        print(f"  Model: {name}  ({len(rows)} version(s))")
        print(f"{'=' * 70}")
        print(f"  {'ID':>4} {'Version':<12} {'Stage':<14} {'Metrics':<30} {'Config'}")
        print(f"  {'-' * 66}")

        for row in rows:
            metrics_str = '  '.join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in json.loads(row['metrics']).items()
            )
            config_str = '  '.join(
                f"{k}={v}"
                for k, v in list(json.loads(row['config']).items())[:3]
            )
            stage_display = f"[{row['stage']}]" if row['stage'] == 'production' else row['stage']
            print(f"  {row['id']:>4} {row['version']:<12} {stage_display:<14} "
                  f"{metrics_str:<30} {config_str}")

        print(f"{'=' * 70}")

    def list_all(self):
        """Print all registered models across all names."""
        conn = _get_conn(self.db_path)
        rows = conn.execute(
            "SELECT * FROM models ORDER BY name, registered_at"
        ).fetchall()
        conn.close()

        if not rows:
            print("No models registered.")
            return

        print(f"\n{'=' * 75}")
        print("  Model Registry — All Models")
        print(f"{'=' * 75}")
        print(f"  {'ID':>4} {'Name':<16} {'Version':<10} {'Stage':<14} {'Metrics'}")
        print(f"  {'-' * 71}")

        for row in rows:
            metrics = json.loads(row['metrics'])
            m_str   = '  '.join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            stage_display = f"[{row['stage']}]" if row['stage'] == 'production' else row['stage']
            print(f"  {row['id']:>4} {row['name']:<16} {row['version']:<10} "
                  f"{stage_display:<14} {m_str}")

        print(f"{'=' * 75}")

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d['metrics'] = json.loads(d['metrics'])
        d['config']  = json.loads(d['config'])
        return d


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, numpy as np

    db  = os.path.join(tempfile.gettempdir(), "test_registry.db")
    mdr = os.path.join(tempfile.gettempdir(), "test_models/")

    if os.path.exists(db):
        os.unlink(db)

    registry = ModelRegistry(db_path=db, model_dir=mdr)

    # Create dummy model files
    for ver, acc in [('v1.0', 0.82), ('v1.1', 0.88), ('v1.2', 0.91)]:
        fpath = os.path.join(tempfile.gettempdir(), f"weights_{ver}.npy")
        import numpy as np
        np.save(fpath, np.zeros(10))

        registry.register('my_mlp', ver, fpath,
                          metrics={'val_acc': acc},
                          config={'hidden': 128, 'lr': 0.01})

    registry.transition_stage('my_mlp', 'v1.0', 'staging')
    registry.transition_stage('my_mlp', 'v1.1', 'production')
    registry.transition_stage('my_mlp', 'v1.2', 'production')  # archives v1.1

    registry.list_versions('my_mlp')

    prod = registry.get_production('my_mlp')
    print(f"\nProduction model: {prod['name']} {prod['version']}  "
          f"val_acc={prod['metrics']['val_acc']:.4f}")

    assert prod['version'] == 'v1.2', "Expected v1.2 to be production"

    v11 = registry.get_model('my_mlp', 'v1.1')
    assert v11['stage'] == 'archived', f"Expected v1.1 to be archived, got {v11['stage']}"

    print("Self-test passed.")

    # Cleanup
    os.unlink(db)
    shutil.rmtree(mdr, ignore_errors=True)
