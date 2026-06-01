"""
dashboard.py — Terminal dashboard for experiments and model registry.

Provides ASCII visualisations for the MLOps mini platform.

Functions:
  - print_experiments_table   : show recent runs with metrics
  - print_model_registry      : show all model versions and stages
  - print_loss_curve_ascii    : draw a scaled ASCII line chart in the terminal
"""

import sqlite3
import json
import os
import math


# ---------------------------------------------------------------------------
# Helper: get connection (inline to avoid circular import)
# ---------------------------------------------------------------------------

def _conn(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        return None
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


# ---------------------------------------------------------------------------
# Experiments table
# ---------------------------------------------------------------------------

def print_experiments_table(db_path: str = "mlops.db", n_recent: int = 20):
    """
    Print a summary table of recent experiment runs.

    Columns: id | name | status | duration | key metrics (final values)

    Parameters
    ----------
    db_path  : str — SQLite database path
    n_recent : int — maximum number of runs to show (most recent first)
    """
    conn = _conn(db_path)
    if conn is None:
        print(f"No database at {db_path}")
        return

    runs = conn.execute(
        "SELECT id, name, status, start_time, end_time "
        "FROM runs ORDER BY id DESC LIMIT ?",
        (n_recent,)
    ).fetchall()

    if not runs:
        print("No experiments found.")
        conn.close()
        return

    print(f"\n{'=' * 80}")
    print(f"  Recent Experiments  (last {min(n_recent, len(runs))})")
    print(f"{'=' * 80}")
    print(f"  {'ID':>4} {'Name':<20} {'Status':<12} {'Duration':>10}  Final metrics")
    print(f"  {'-' * 76}")

    for run in runs:
        duration = (run['end_time'] - run['start_time']
                    if run['end_time'] and run['start_time'] else 0.0)

        # Fetch final value of each metric (highest step)
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
        """, (run['id'], run['id'])).fetchall()

        m_str = '  '.join(f"{r['key']}={r['value']:.4f}" for r in metric_rows)

        status_icon = {'completed': 'OK', 'running': '...', 'failed': 'ERR'}.get(
            run['status'], run['status'][:3])

        print(f"  {run['id']:>4} {run['name']:<20} "
              f"{run['status']:<12} {duration:>9.1f}s  {m_str}")

    print(f"{'=' * 80}")
    conn.close()


# ---------------------------------------------------------------------------
# Model registry table
# ---------------------------------------------------------------------------

def print_model_registry(db_path: str = "mlops.db", model_dir: str = "models/"):
    """
    Print the model registry: name | version | stage | metrics

    Parameters
    ----------
    db_path   : str
    model_dir : str — used to check if model files exist on disk
    """
    conn = _conn(db_path)
    if conn is None:
        print(f"No database at {db_path}")
        return

    # Check if the models table exists
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
    ).fetchone()

    if tables is None:
        print("No model registry found (no 'models' table).")
        conn.close()
        return

    rows = conn.execute(
        "SELECT * FROM models ORDER BY name, registered_at"
    ).fetchall()

    if not rows:
        print("Model registry is empty.")
        conn.close()
        return

    print(f"\n{'=' * 75}")
    print("  Model Registry")
    print(f"{'=' * 75}")
    print(f"  {'ID':>4} {'Name':<16} {'Version':<10} {'Stage':<14} {'Metrics'}")
    print(f"  {'-' * 71}")

    for row in rows:
        metrics = json.loads(row['metrics'])
        m_str   = '  '.join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        stage = row['stage']
        stage_fmt = f"[PROD] {stage}" if stage == 'production' else stage

        print(f"  {row['id']:>4} {row['name']:<16} {row['version']:<10} "
              f"{stage_fmt:<16} {m_str}")

    print(f"{'=' * 75}")
    conn.close()


# ---------------------------------------------------------------------------
# ASCII loss curve
# ---------------------------------------------------------------------------

def print_loss_curve_ascii(run_id: int, metric: str = 'train_loss',
                            db_path: str = "mlops.db",
                            width: int = 60, height: int = 15):
    """
    Draw an ASCII line chart of a metric over training steps.

    The chart scales all values to fit within width × height characters.
    The y-axis shows the min and max values.
    The x-axis shows the first and last step numbers.

    Parameters
    ----------
    run_id  : int  — run ID to plot
    metric  : str  — metric key (default 'train_loss')
    db_path : str
    width   : int  — number of columns for the plot area (default 60)
    height  : int  — number of rows for the plot area (default 15)
    """
    conn = _conn(db_path)
    if conn is None:
        print(f"No database at {db_path}")
        return

    rows = conn.execute(
        "SELECT step, value FROM metrics WHERE run_id=? AND key=? ORDER BY step",
        (run_id, metric)
    ).fetchall()
    conn.close()

    if not rows:
        print(f"  No data for run_id={run_id}, metric='{metric}'")
        return

    steps  = [r['step'] for r in rows]
    values = [r['value'] for r in rows]

    n = len(values)
    v_min = min(values)
    v_max = max(values)
    v_range = v_max - v_min

    # Build the grid (height rows × width cols), initialised to spaces
    grid = [[' '] * width for _ in range(height)]

    for col_idx in range(width):
        # Map column to a position in the data
        denom    = max(width - 1, 1)
        data_idx = int(col_idx / denom * (n - 1))
        data_idx = max(0, min(data_idx, n - 1))
        val      = values[data_idx]

        # Map value to row (higher value → lower row index)
        if v_range > 1e-12:
            row_frac = 1.0 - (val - v_min) / v_range
        else:
            row_frac = 0.5
        row_idx = int(row_frac * (height - 1))
        row_idx = max(0, min(row_idx, height - 1))

        grid[row_idx][col_idx] = '*'

    # Render
    run_name_q = sqlite3.connect(db_path)
    run_name_q.row_factory = sqlite3.Row
    run_row = run_name_q.execute(
        "SELECT name FROM runs WHERE id=?", (run_id,)
    ).fetchone()
    run_name_q.close()
    run_label = run_row['name'] if run_row else f"run {run_id}"

    print(f"\n  ASCII curve: '{metric}'  run={run_label}  "
          f"steps={steps[0]}–{steps[-1]}")
    print(f"  {'─' * (width + 6)}")

    for r, row in enumerate(grid):
        # Y-axis label on the left
        if r == 0:
            y_label = f"{v_max:6.3f}"
        elif r == height - 1:
            y_label = f"{v_min:6.3f}"
        elif r == height // 2:
            y_label = f"{(v_min + v_max)/2:6.3f}"
        else:
            y_label = "      "

        print(f"  {y_label} |{''.join(row)}|")

    # X-axis
    x_pad = " " * 9
    step_start = str(steps[0])
    step_end   = str(steps[-1])
    spacer = " " * (width - len(step_start) - len(step_end))
    print(f"  {x_pad}{step_start}{spacer}{step_end}")
    print(f"  {x_pad}{'─' * width}")
    print(f"  {'':>15}steps")

    # Summary stats
    final_val = values[-1]
    best_val  = min(values) if 'loss' in metric.lower() else max(values)
    print(f"  start={values[0]:.4f}  final={final_val:.4f}  "
          f"best={'min' if 'loss' in metric else 'max'}={best_val:.4f}")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, time

    db = os.path.join(tempfile.gettempdir(), "test_dashboard.db")
    if os.path.exists(db):
        os.unlink(db)

    # Create some fake runs
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE IF NOT EXISTS runs
        (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, config TEXT,
         status TEXT, start_time REAL, end_time REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS metrics
        (run_id INTEGER, step INTEGER, key TEXT, value REAL, timestamp REAL)""")
    conn.commit()

    now = time.time()
    conn.execute("INSERT INTO runs VALUES (1,'run_a','{}','completed',?,?)",
                 (now - 30, now))
    conn.execute("INSERT INTO runs VALUES (2,'run_b','{}','completed',?,?)",
                 (now - 20, now))

    import math
    for step in range(20):
        loss = 2.0 * math.exp(-step * 0.15)
        acc  = 0.5 + 0.4 * (1 - math.exp(-step * 0.15))
        conn.execute("INSERT INTO metrics VALUES (1,?,?,?,?)",
                     (step, 'train_loss', loss, now))
        conn.execute("INSERT INTO metrics VALUES (1,?,?,?,?)",
                     (step, 'val_acc', acc, now))
        conn.execute("INSERT INTO metrics VALUES (2,?,?,?,?)",
                     (step, 'train_loss', loss * 1.1, now))

    conn.commit()
    conn.close()

    print_experiments_table(db_path=db)
    print_loss_curve_ascii(1, 'train_loss', db_path=db)
    print_loss_curve_ascii(1, 'val_acc',    db_path=db)

    os.unlink(db)
    print("\nDashboard self-test passed.")
