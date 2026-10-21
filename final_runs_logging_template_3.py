# Databricks notebook source
# MAGIC %md
# MAGIC # Final runs — RF / LogReg / xgboost on Train Set A, evaluated on Test Set A
# MAGIC
# MAGIC **AI-assisted:** generic template drafted by Claude (Anthropic), 2026-10-21 (rev. 2). Cite per the
# MAGIC course AI Use Policy and note which cells you used as-is vs. modified.
# MAGIC
# MAGIC **Assumes these already exist from your current notebook** (Train Set A–TFIDF fit on Train Set A only):
# MAGIC - `X_train` (sparse hstack of the 7 `fit_transform` outputs), `y_train` (string labels, no nulls)
# MAGIC - `X_test` (sparse hstack of the 7 `transform` outputs, same vectorizers, same feature order), `y_test` (string labels, no nulls)
# MAGIC - `test_feature_key` — the `feature_key` column of the test table, aligned row-for-row with `X_test`
# MAGIC - `vocab`, `vectorizers` — the fitted Train Set A–TFIDF objects (optional; set to `None` if not collected yet)
# MAGIC - for the PROD cell: `prod_model`, `X_test_prod` (Test Set A transformed with PROD vocab + PROD TFIDF)
# MAGIC
# MAGIC **Test Set A here excludes out-of-taxonomy (null-label) rows** — removed before the test table was built.
# MAGIC All accuracies below are therefore "in-taxonomy accuracy". The OTHER version is done separately later.
# MAGIC
# MAGIC Placeholders in `<angle brackets>` must be filled in.

# COMMAND ----------

# 1. Fixed settings — decided BEFORE running; do not change after seeing any test result
import json, os, tempfile, time
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

CLASSES = ["apparel", "bags", "accessories", "footwear", "watches", "jewellery", "beauty"]  # must match label strings exactly
SEED = 42
CV_FOLDS = 5
CV_SCORING = "f1_macro"          # DECIDED 2026-10-21 (PROGRESS #41)
CONF_CUT = 0.5
EXPERIMENT_PATH = None           # None = log to this notebook's own experiment (Databricks default).
                                 # To use a specific experiment, paste its path from the experiment page.
PRED_TABLE = "<catalog>.<schema>.test_predictions"
TRAIN_SET, TRAIN_SET_VERSION = "TrainSetA", "<delta_version>"
TEST_SET = "A"
PREPROCESSING = "TrainSetA-TFIDF"
PREPROCESSING_OBJECTS = {"vocab": vocab, "vectorizers": vectorizers}   # or None to skip saving them

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
test_feature_key = np.asarray(test_feature_key)
assert set(np.unique(y_train)) <= set(CLASSES), "unexpected training labels"
assert set(np.unique(y_test)) <= set(CLASSES), "unexpected test labels (nulls/OTHER should not be in this test table)"
assert X_test.shape[0] == len(y_test) == len(test_feature_key), "X_test, y_test, feature_key not aligned"
assert len(np.unique(test_feature_key)) == len(test_feature_key), "feature_key not unique in test"

if EXPERIMENT_PATH:
    mlflow.set_experiment(EXPERIMENT_PATH)
# autolog: params, best params, cv_results, child runs, the fitted model. Ignore its *training* metrics.
mlflow.sklearn.autolog(log_models=True, max_tuning_runs=None, silent=True)
try:
    mlflow.xgboost.autolog(disable=True)   # avoid duplicate runs when XGBClassifier sits inside GridSearchCV
except Exception:
    pass

# COMMAND ----------

# 2. Model configs. Fits per config = n_combinations x CV_FOLDS (+1 refit).
#    Time ONE fit first and multiply before launching.
GRIDS = {
    # PROD recipe on new data: sklearn defaults + n_jobs=32 (PROGRESS #42). One combination, no tuning.
    # Only difference from PROD: random_state fixed (PROD used None) — affects reproducibility, not the recipe.
    # gs_n_jobs=1 because the RF itself already uses 32 cores.
    "rf_prodcfg": dict(
        estimator=RandomForestClassifier(n_jobs=32, random_state=SEED),
        grid={"n_estimators": [100]},          # = the default; written out so the grid isn't empty
        encode_labels=False,
        gs_n_jobs=1,
    ),
    "rf": dict(
        estimator=RandomForestClassifier(random_state=SEED, n_jobs=1),
        grid={
            "n_estimators": [300],
            "max_features": ["sqrt"],
            "min_samples_leaf": [1, 2],
            "class_weight": [None, "balanced_subsample"],
        },
        encode_labels=False,
    ),
    "logreg": dict(
        estimator=LogisticRegression(solver="saga", max_iter=3000, random_state=SEED),
        grid={"C": [0.3, 1, 3, 10], "class_weight": [None, "balanced"]},
        encode_labels=False,
    ),
    "xgb": dict(
        estimator=XGBClassifier(
            objective="multi:softprob", tree_method="hist", n_estimators=400, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=SEED, n_jobs=1,
        ),
        grid={"max_depth": [4, 6, 8], "min_child_weight": [1, 3]},
        encode_labels=True,   # xgboost needs integer labels; mapped back below
    ),
}
for k, v in GRIDS.items():
    n = int(np.prod([len(x) for x in v["grid"].values()]))
    print(f"{k}: {n} combinations x {CV_FOLDS} folds = {n * CV_FOLDS} fits (+1 refit)")

# COMMAND ----------

# 3. Metric helpers — used for every model AND for PROD, so all rows are like-for-like
def compute_test_metrics(y_true, y_pred, conf):
    y_true, y_pred, conf = np.asarray(y_true), np.asarray(y_pred), np.asarray(conf)
    keep = conf >= CONF_CUT
    m = {
        "test_n": len(y_true),
        "test_acc": accuracy_score(y_true, y_pred),                            # headline (in-taxonomy)
        "test_f1_macro": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "test_f1_weighted": f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0),
        "test_coverage_conf05": float(keep.mean()),                            # always report with acc_conf05
        "test_acc_conf05": accuracy_score(y_true[keep], y_pred[keep]) if keep.any() else np.nan,
    }
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=CLASSES),
                      index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])
    return {k: float(v) for k, v in m.items()}, report, cm


def log_test_outputs(run_id, model_name, y_pred, conf, tmpdir):
    """Logs test metrics/report/confusion matrix to the active run and appends per-record rows to PRED_TABLE."""
    metrics, report, cm = compute_test_metrics(y_test, y_pred, conf)
    mlflow.log_metrics(metrics)
    p = os.path.join(tmpdir, "test_report.json")
    with open(p, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(p, "test")
    p = os.path.join(tmpdir, "confusion_matrix.csv"); cm.to_csv(p); mlflow.log_artifact(p, "test")

    preds = pd.DataFrame({
        "run_id": run_id, "model": model_name, "test_set": TEST_SET,
        "feature_key": test_feature_key,
        "y_true": y_test, "y_pred": np.asarray(y_pred), "conf": np.asarray(conf, dtype=float),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    p = os.path.join(tmpdir, "test_predictions.csv"); preds.to_csv(p, index=False); mlflow.log_artifact(p, "test")
    spark.createDataFrame(preds).write.format("delta").mode("append").saveAsTable(PRED_TABLE)
    return metrics

# COMMAND ----------

# 4. One final run per config: tune by CV on Train Set A only, then ONE evaluation on Test Set A
def run_final(model_name, estimator, grid, encode_labels=False, gs_n_jobs=-1):
    le = LabelEncoder().fit(CLASSES) if encode_labels else None
    y_fit = le.transform(y_train) if le is not None else y_train
    gs = GridSearchCV(estimator, grid, scoring=CV_SCORING, cv=cv, n_jobs=gs_n_jobs, refit=True)

    with mlflow.start_run(run_name=f"{model_name}_final_{TRAIN_SET}") as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags({
            "model_family": model_name, "train_set": TRAIN_SET, "train_set_version": TRAIN_SET_VERSION,
            "test_set": TEST_SET, "test_other_rows": "excluded", "preprocessing": PREPROCESSING,
            "stage": "final",
        })
        mlflow.log_params({"cv_folds": CV_FOLDS, "cv_scoring": CV_SCORING, "seed": SEED,
                           "conf_cut": CONF_CUT, "grid": json.dumps(grid, default=str)})

        t0 = time.perf_counter(); gs.fit(X_train, y_fit); grid_seconds = time.perf_counter() - t0
        best = gs.best_estimator_

        # CV = tuning only, never reported as performance (#36)
        mlflow.log_metrics({
            "cv_best_mean": float(gs.cv_results_["mean_test_score"][gs.best_index_]),
            "cv_best_std": float(gs.cv_results_["std_test_score"][gs.best_index_]),
        })
        p = os.path.join(tmp, "cv_results.csv"); pd.DataFrame(gs.cv_results_).to_csv(p, index=False)
        mlflow.log_artifact(p, "cv")

        t0 = time.perf_counter(); proba = best.predict_proba(X_test); predict_seconds = time.perf_counter() - t0
        labels = best.classes_ if le is None else le.inverse_transform(best.classes_)
        y_pred = np.asarray(labels)[proba.argmax(axis=1)]
        conf = proba.max(axis=1)

        mlflow.log_metrics({
            "grid_seconds": grid_seconds, "refit_seconds": float(gs.refit_time_),
            "predict_seconds": predict_seconds, "predict_rows_per_sec": len(y_pred) / predict_seconds,
        })
        # Preprocessing objects are part of the model version (#36)
        if PREPROCESSING_OBJECTS is not None:
            p = os.path.join(tmp, "preprocessing.joblib")
            joblib.dump(PREPROCESSING_OBJECTS, p)
            mlflow.log_artifact(p, "preprocessing")

        metrics = log_test_outputs(run.info.run_id, model_name, y_pred, conf, tmp)
        mlflow.set_tag("n_test_evaluations_this_config", "1")
        print(model_name, "best params:", gs.best_params_)
        print({k: round(v, 4) for k, v in metrics.items()})
        return gs, run.info.run_id

# COMMAND ----------

# 5. Launch (one config per cell so a failure doesn't lose the others)
gs_rf_prodcfg, run_rf_prodcfg = run_final("rf_prodcfg", **GRIDS["rf_prodcfg"])

# COMMAND ----------

gs_rf, run_rf = run_final("rf", **GRIDS["rf"])

# COMMAND ----------

gs_lr, run_lr = run_final("logreg", **GRIDS["logreg"])

# COMMAND ----------

gs_xgb, run_xgb = run_final("xgb", **GRIDS["xgb"])

# COMMAND ----------

# 6. PROD reference rows — the actual PROD model (trained on Legacy) with PROD TFIDF; same metrics, same table
mlflow.sklearn.autolog(disable=True)
with mlflow.start_run(run_name="PROD_reference_testA") as run, tempfile.TemporaryDirectory() as tmp:
    mlflow.set_tags({"model_family": "rf", "train_set": "Legacy", "test_set": TEST_SET,
                     "test_other_rows": "excluded", "preprocessing": "PROD-TFIDF", "stage": "reference"})
    t0 = time.perf_counter(); proba = prod_model.predict_proba(X_test_prod); predict_seconds = time.perf_counter() - t0
    y_pred = np.asarray(prod_model.classes_)[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    mlflow.log_metrics({"predict_seconds": predict_seconds, "predict_rows_per_sec": len(y_pred) / predict_seconds})
    log_test_outputs(run.info.run_id, "PROD", y_pred, conf, tmp)
    run_prod = run.info.run_id
mlflow.sklearn.autolog(log_models=True, max_tuning_runs=None, silent=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation — report-ready tables (run after sections 5–6; reads only what was logged)

# COMMAND ----------

# 7a. Model comparison table (one row per final/reference run)
FINAL_RUNS = {"PROD": run_prod, "rf_prodcfg": run_rf_prodcfg, "rf": run_rf,
              "logreg": run_lr, "xgb": run_xgb}   # or paste run_ids from the MLflow page
cols = ["test_n", "test_acc", "test_f1_macro", "test_f1_weighted", "test_acc_conf05", "test_coverage_conf05",
        "cv_best_mean", "cv_best_std", "refit_seconds", "predict_rows_per_sec"]
rows = []
for name, rid in FINAL_RUNS.items():
    r = mlflow.get_run(rid).data.metrics
    rows.append({"model": name, **{c: r.get(c, np.nan) for c in cols}})
comparison = pd.DataFrame(rows).set_index("model")
display(comparison.round(4))

# COMMAND ----------

# 7b. Per-class table with 95% Wilson intervals (recall: denominator = true class; precision: = predicted class)
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)

preds = spark.table(PRED_TABLE).filter(f"run_id IN ({','.join(repr(r) for r in FINAL_RUNS.values())})").toPandas()
per_class = []
for name, rid in FINAL_RUNS.items():
    d = preds[preds.run_id == rid]
    for c in CLASSES:
        tp = int(((d.y_true == c) & (d.y_pred == c)).sum())
        n_true = int((d.y_true == c).sum()); n_pred = int((d.y_pred == c).sum())
        rl, rh = wilson(tp, n_true); pl, ph = wilson(tp, n_pred)
        per_class.append({"model": name, "class": c, "support": n_true,
                          "recall": tp / n_true if n_true else np.nan, "recall_lo": rl, "recall_hi": rh,
                          "precision": tp / n_pred if n_pred else np.nan, "precision_lo": pl, "precision_hi": ph})
per_class = pd.DataFrame(per_class)
display(per_class.round(3))

# COMMAND ----------

# 7c. Paired comparison vs PROD on the SAME test records: bootstrap CI on the accuracy difference + exact McNemar
def paired_compare(correct_a, correct_b, n_boot=2000, seed=SEED):
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    only_a = int(((a == 1) & (b == 0)).sum()); only_b = int(((a == 0) & (b == 1)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": a.mean() - b.mean(), "ci_lo": lo, "ci_hi": hi,
            "a_right_b_wrong": only_a, "a_wrong_b_right": only_b, "mcnemar_p": p}

wide = (preds.assign(correct=(preds.y_true == preds.y_pred).astype(int))
             .pivot_table(index="feature_key", columns="model", values="correct"))
assert wide.notna().all().all() and len(wide) == len(y_test), "record sets differ between runs"
paired = pd.DataFrame({m: paired_compare(wide[m], wide["PROD"])
                       for m in ["rf_prodcfg", "rf", "logreg", "xgb"]}).T
display(paired.round(4))
# Apply the decision rule fixed in PROGRESS.md (CI excludes 0 AND gain >= threshold) — written down before this cell is run.
