# MLOps Mini

A minimal MLOps platform built with Python stdlib only (SQLite, json, shutil).
Covers experiment tracking, model versioning, and a terminal dashboard.

## Project layout

```
mlops-mini/
├── src/
│   ├── tracker.py        # Experiment tracker backed by SQLite
│   ├── model_registry.py # Model versioning with stage transitions
│   ├── dashboard.py      # Terminal tables + ASCII loss curves
│   └── demo.py           # End-to-end demo: 5 runs → register → dashboard
├── README.md
├── LEARNINGS.md
└── requirements.txt
```

## Quick start

```bash
cd mlops-mini
pip install -r requirements.txt   # only numpy needed
python src/demo.py
```

## What the demo does

1. Trains 5 MLP configurations with different learning rates and hidden sizes.
2. Logs `train_loss`, `train_acc`, `val_acc` every 10 epochs to SQLite.
3. Registers the best-performing model in the model registry.
4. Demonstrates stage transitions: development → staging → production.
5. Shows a terminal dashboard with experiment comparison and ASCII loss curves.

## Core APIs

### Experiment Tracker

```python
from tracker import Experiment, list_experiments, compare_experiments

exp = Experiment('my_run', config={'lr': 0.01, 'hidden': 128}).start()

for epoch in range(100):
    exp.log(epoch, {'train_loss': 0.4, 'val_acc': 0.88})

exp.finish()

list_experiments()
compare_experiments([1, 2, 3], metric='val_acc')
```

### Model Registry

```python
from model_registry import ModelRegistry

reg = ModelRegistry()
reg.register('my_mlp', 'v1.0', '/path/to/weights.npy',
             metrics={'val_acc': 0.92},
             config={'hidden': 128})

reg.transition_stage('my_mlp', 'v1.0', 'staging')
reg.transition_stage('my_mlp', 'v1.0', 'production')

prod = reg.get_production('my_mlp')   # → {'version': 'v1.0', ...}
reg.list_versions('my_mlp')
```

### Dashboard

```python
from dashboard import (print_experiments_table, print_model_registry,
                        print_loss_curve_ascii)

print_experiments_table()
print_loss_curve_ascii(run_id=3, metric='train_loss')
print_model_registry()
```

## Sample output

```
==========================================================================
  Recent Experiments
==========================================================================
  ID Name                 Status       Duration  Final metrics
  -----------------------------------------------------------------------
   1 run_lr001            completed       4.2s  train_loss=1.2341  val_acc=0.7812
   2 run_lr003            completed       4.1s  train_loss=0.9812  val_acc=0.8234
   3 run_lr01             completed       4.3s  train_loss=0.8901  val_acc=0.8456
  ...

  ASCII curve: 'train_loss'  run=run_lr01  steps=10–80
  ────────────────────────────────────────────────────
  2.3141 |****                                                       |
         |    ***                                                    |
         |       **                                                  |
  1.2000 |         **                                                |
         |           ***                                             |
  0.8901 |              *************************************        |
         10                                                  80
         ─────────────────────────────────────────────────────
```
