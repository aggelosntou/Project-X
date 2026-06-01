# Learnings — MLOps Mini

## 1. Why experiment tracking matters

### The problem without tracking

Without a tracker, a research workflow looks like:
- Run experiment, write accuracy in a text file or notebook cell.
- Change hyperparameter, re-run, update the number.
- Come back a week later: which config produced which result?

This makes it impossible to:
- Reproduce results (exact hyperparameters forgotten)
- Compare systematically (different configs run at different times)
- Catch regressions (the new model might be worse on a different metric)

### What tracking gives you

Every experiment records:
1. **Config**: exact hyperparameters (`lr`, `hidden_size`, `batch_size`, ...)
2. **Metrics over time**: `val_acc` at every epoch, not just the final value
3. **Status and duration**: did it finish? how long?
4. **Reproducibility**: given a run ID, you can re-run with identical settings

This is why tools like MLflow, Weights & Biases, and Neptune exist —
they are databases for experiment metadata.

---

## 2. What model versioning prevents

### Accidental deployment

Without versioning, a common failure mode:
1. You retrain a model and save it as `model.pkl`.
2. You deploy it to production.
3. Two weeks later you realise the new model has a data-preprocessing bug.
4. You cannot easily roll back because the old weights were overwritten.

Model versioning stores every version with:
- Its exact weights file (copy in a registry directory)
- Its metrics at registration time
- Its current lifecycle stage (development / staging / production / archived)

### The stage model

```
development → staging → production → archived
```

- **development**: freshly trained, not yet validated
- **staging**: validated offline; ready for canary/shadow testing
- **production**: currently serving live traffic
- **archived**: was production, replaced by a newer version

Promoting a new model to **production** automatically archives the previous one.
This means:
- You can always find the last-known-good model (latest archived)
- Rollback = promote the archived version back to production
- No file is ever deleted

---

## 3. What data drift is

**Data drift** (also called covariate shift or dataset shift) occurs when the
statistical distribution of inputs in production changes over time compared to
the training distribution.

### Example

A fraud detection model is trained on transactions from January.
In December, spending patterns change (Christmas shopping).
Features like `transaction_amount` and `merchant_category` now have different
means and variances.  The model's decision boundary, tuned for January data,
is now poorly aligned with December data.

Symptoms:
- Accuracy/recall on live traffic degrades even though the model hasn't changed
- Feature importance shifts (features that mattered in January now don't)

Detection:
- Monitor rolling statistics of input features (mean, std, quantiles)
- Use population stability index (PSI) or KL divergence to compare
  training distribution vs recent production distribution
- Alert when PSI > 0.2 (conventional threshold for significant drift)

---

## 4. Staging vs production

| Stage | Meaning |
|-------|---------|
| staging | Model has passed offline evaluation; serving shadow traffic or canary (1–5%) |
| production | Model is serving 100% of live traffic |

**Why stage before production?**

Shadow deployment: the staging model receives the same requests as production
but its outputs are discarded.  You can compare the two models' outputs and
check that the new model agrees with the old one on ≥95% of requests before
flipping traffic.

Canary deployment: route 1–5% of traffic to the new model.  Monitor error
rates and latency.  If metrics look good after a day, gradually ramp up to
100%.  This limits blast radius if the new model has a bug.

---

## 5. The feature store concept

A **feature store** is a centralised repository of precomputed features
shared across multiple models and experiments.

### The problem it solves

Without a feature store:
- Team A computes "user 7-day purchase history" in their training pipeline.
- Team B computes the same feature in their pipeline.
- The two computations are slightly different (different SQL, different time zones).
- At inference time, the feature is computed a third way.

This training/serving skew causes silent accuracy degradation.

### What a feature store provides

1. **Feature definitions**: compute once, reuse everywhere
2. **Offline store**: historical features for training (e.g. BigQuery / Hive)
3. **Online store**: low-latency feature lookup for inference (e.g. Redis / DynamoDB)
4. **Point-in-time correctness**: when training on historical data, features are
   looked up at the time the label was generated — not at the current time
   (prevents data leakage)

Popular feature stores: Feast, Tecton, Vertex AI Feature Store, Databricks.
The SQLite database in this project is a toy approximation of the offline store.

---

## 6. Why SQLite is enough for small teams

SQLite is:
- A single file on disk (no server process)
- ACID-compliant (transactions are safe)
- Fast enough for hundreds of experiment runs
- Zero operational overhead

At scale, a team would migrate to:
- **PostgreSQL**: for concurrent writes from multiple training machines
- **Parquet + S3**: for metric storage at billions of rows
- **Dedicated MLOps platform**: MLflow, W&B, Neptune, Vertex AI

The design of this mini tracker mirrors those platforms' schemas exactly:
`runs` table + `metrics` table with foreign key.  Migrating to MLflow
would mean swapping `sqlite3` calls for `mlflow.log_metric()`.
