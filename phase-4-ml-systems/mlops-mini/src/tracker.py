"""
tracker.py — Minimal experiment tracker backed by SQLite.

Provides:
  - Experiment : context-manager that logs metrics to a local SQLite DB
  - list_experiments  : print all runs with their final metrics
  - compare_experiments : side-by-side comparison of selected runs

Design
------
Every run is stored in two tables:
  runs    : id, name, config (JSON), status, start_time, end_time
  metrics : run_id, step, key, value, timestamp

This mirrors what MLflow, W&B, and Neptune do, except they add:
  - Artifact storage (model files, plots)
  - Remote sync / UI
  - Team access control
"""

import sqlite3
import json
import time
import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

DDL_RUNS = """
    CREATE TABLE IF NOT EXISTS runs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT    NOT NULL,
        config     TEXT    NOT NULL DEFAULT '{}',
        status     TEXT    NOT NULL DEFAULT 'running',
        start_time REAL,
        end_time   REAL
    )
"""

DDL_METRICS = """
    CREATE TABLE IF NOT EXISTS metrics (
        run_id    INTEGER NOT NULL,
        step      INTEGER NOT NULL,
        key       TEXT    NOT NULL,
        value     REAL    NOT NULL,
        timestamp REAL    NOT NULL
    )
"""


def _get_conn(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure tables exist."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(DDL_RUNS)
    conn.execute(DDL_METRICS)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """
    Track a single training run.

    Usage
    -----
        exp = Experiment('my_run', {'lr': 0.01, 'hidden': 64}).start()
        for step in range(100):
            exp.log(step, {'train_loss': 1.5, 'val_acc': 0.82})
        exp.finish()

    Parameters
    ----------
    name    : str  — human-readable run name
    config  : dict — hyperparameters and other metadata
    db_path : str  — path to the SQLite database file
    """

    def __init__(self, name: str, config: dict, db_path: str = "mlops.db"):
        self.name     = name
        self.config   = config
        self.db_path  = db_path
        self.run_id   = None
        self._conn    = None
        self._start_time = None

    def start(self) -> "Experiment":
        """
        Create the run record in the database.
        Returns self for chaining: exp = Experiment(...).start()
        """
        self._conn       = _get_conn(self.db_path)
        self._start_time = time.time()

        cur = self._conn.execute(
            "INSERT INTO runs (name, config, status, start_time) VALUES (?,?,?,?)",
            (self.name, json.dumps(self.config), 'running', self._start_time)
        )
        self.run_id = cur.lastrowid
        self._conn.commit()
        return self

    def log(self, step: int, metrics: dict):
        """
        Log a dict of metric key→value at a given training step.

        Parameters
        ----------
        step    : int  — training step / epoch number
        metrics : dict — e.g. {'train_loss': 0.42, 'val_acc': 0.91}
        """
        if self._conn is None:
            raise RuntimeError("Call .start() before .log()")

        t    = time.time()
        rows = [
            (self.run_id, step, str(k), float(v), t)
            for k, v in metrics.items()
        ]
        self._conn.executemany(
            "INSERT INTO metrics (run_id, step, key, value, timestamp) VALUES (?,?,?,?,?)",
            rows
        )
        self._conn.commit()

    def finish(self, status: str = 'completed'):
        """
        Mark the run as finished and close the database connection.

        Parameters
        ----------
        status : str — one of: 'completed', 'failed', 'killed'
        """
        if self._conn is None:
            return

        self._conn.execute(
            "UPDATE runs SET status=?, end_time=? WHERE id=?",
            (status, time.time(), self.run_id)
        )
        self._conn.commit()
        self._conn.close()
        self._conn = None

    def duration(self) -> float:
        """Elapsed time since .start() in seconds."""
        return time.time() - self._start_time if self._start_time else 0.0


# ---------------------------------------------------------------------------
# list_experiments
# ---------------------------------------------------------------------------

def list_experiments(db_path: str = "mlops.db"):
    """
    Print a table of all experiment runs with their final logged metrics.

    Shows: id | name | status | duration | last logged metrics
    """
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        return

    conn = _get_conn(db_path)

    runs = conn.execute(
        "SELECT id, name, config, status, start_time, end_time "
        "FROM runs ORDER BY id"
    ).fetchall()

    if not runs:
        print("No experiments recorded.")
        conn.close()
        return

    print(f"\n{'=' * 80}")
    print(f"  Experiment Runs  ({db_path})")
    print(f"{'=' * 80}")
    print(f"  {'ID':>4} {'Name':<20} {'Status':<12} {'Duration':>10}  Latest metrics")
    print(f"  {'-' * 76}")

    for run in runs:
        run_id   = run['id']
        duration = (run['end_time'] - run['start_time']
                    if run['end_time'] and run['start_time']
                    else 0.0)

        # Get the latest value for each metric key (highest step)
        metric_rows = conn.execute("""
            SELECT m.key, m.value
            FROM metrics m
            INNER JOIN (
                SELECT key, MAX(step) AS max_step
                FROM metrics WHERE run_id = ?
                GROUP BY key
            ) latest ON m.key = latest.key AND m.step = latest.max_step
            WHERE m.run_id = ?
            ORDER BY m.key
        """, (run_id, run_id)).fetchall()

        metric_str = '  '.join(
            f"{r['key']}={r['value']:.4f}" for r in metric_rows
        )

        print(f"  {run_id:>4} {run['name']:<20} {run['status']:<12} "
              f"{duration:>9.1f}s  {metric_str}")

    print(f"{'=' * 80}")
    conn.close()


# ---------------------------------------------------------------------------
# compare_experiments
# ---------------------------------------------------------------------------

def compare_experiments(run_ids: list, metric: str = 'val_acc',
                         db_path: str = "mlops.db"):
    """
    Print a comparison table for the specified runs.

    Shows: id | name | config summary | best metric value | final metric value

    Parameters
    ----------
    run_ids : list[int] — run IDs to compare
    metric  : str       — metric key to compare (default 'val_acc')
    db_path : str
    """
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}")
        return

    conn = _get_conn(db_path)

    print(f"\n{'=' * 72}")
    print(f"  Experiment Comparison — metric: '{metric}'")
    print(f"{'=' * 72}")
    print(f"  {'ID':>4} {'Name':<20} {'Config':<20} {'Best':>8} {'Final':>8}")
    print(f"  {'-' * 68}")

    rows_data = []
    for rid in run_ids:
        run = conn.execute(
            "SELECT id, name, config FROM runs WHERE id=?", (rid,)
        ).fetchone()

        if run is None:
            print(f"  {rid:>4} (not found)")
            continue

        vals = conn.execute(
            "SELECT value FROM metrics WHERE run_id=? AND key=? ORDER BY step",
            (rid, metric)
        ).fetchall()

        values = [r['value'] for r in vals]
        best   = max(values) if values else float('nan')
        final  = values[-1]  if values else float('nan')

        # Summarise config
        config = json.loads(run['config'])
        cfg_str = '  '.join(f"{k}={v}" for k, v in list(config.items())[:3])

        rows_data.append((rid, run['name'], cfg_str, best, final))
        print(f"  {rid:>4} {run['name']:<20} {cfg_str:<20} {best:>8.4f} {final:>8.4f}")

    # Highlight best
    if rows_data:
        best_row = max(rows_data, key=lambda r: r[3])
        print(f"\n  Best run: id={best_row[0]}  name={best_row[1]}  "
              f"best_{metric}={best_row[3]:.4f}")

    print(f"{'=' * 72}")
    conn.close()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    db = os.path.join(tempfile.gettempdir(), "test_tracker.db")
    if os.path.exists(db):
        os.unlink(db)

    # Run 3 quick experiments
    for i, lr in enumerate([0.001, 0.01, 0.1]):
        exp = Experiment(f"run_{i}", {"lr": lr, "hidden": 64}, db_path=db).start()
        for step in range(5):
            exp.log(step, {"train_loss": 2.0 - step * lr * 10,
                            "val_acc":   0.5 + step * lr})
        exp.finish()

    list_experiments(db_path=db)
    compare_experiments([1, 2, 3], metric='val_acc', db_path=db)

    os.unlink(db)
    print("\nSelf-test passed.")
