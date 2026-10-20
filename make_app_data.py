"""Rebuild the app's data/ folder from the project notebook.

The data/ folder is already built, so you only need this if you change the notebook. It runs the notebook
unchanged, adds one export cell at the end, and that cell writes the fitted model parameters, every
customer's cluster labels and the result tables into data/. The cell also checks that the exported
parameters reproduce every cluster label exactly, so the app and the notebook always agree.

Usage:
    python make_app_data.py --notebook "../Customer_Segmentation_CLV_Comparative_Clustering.ipynb"
                            --excel "Online Retail.xlsx"

It needs the notebook's libraries plus nbclient and ipykernel (see requirements-build.txt).
"""
import argparse
import shutil
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXPORT_CELL = r'''
# =====================================================================
# 29.1  EXPORT FOR THE WEB APP (model parameters, labels and result tables)
# =====================================================================
APP_DATA = Path(__APP_DATA_DIR__)
APP_DATA.mkdir(parents=True, exist_ok=True)

def py(value):
    """Convert numpy / pandas values to plain Python so they can be written as JSON."""
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, np.ndarray):
        return py(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value

def raw_to_final(raw_labels, final_labels):
    """Mapping from the model's own cluster numbers to the final (Monetary-ordered) numbers."""
    mapping = {}
    for raw, final in zip(np.asarray(raw_labels), np.asarray(final_labels)):
        mapping.setdefault(int(raw), int(final))
    return mapping

def reorder(rows_by_raw, mapping):
    ordered = [None] * len(mapping)
    for raw, final in mapping.items():
        ordered[final] = np.asarray(rows_by_raw[raw], dtype=float)
    return np.array(ordered)

def nearest(X, centers):
    return (((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)).argmin(axis=1)

def gmm_predict(X, weights, means, precisions_cholesky):
    """Same formula as GaussianMixture.predict (covariance_type='full')."""
    n_features = X.shape[1]
    log_prob = np.empty((X.shape[0], len(weights)))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_cholesky)):
        y = X @ prec_chol - mu @ prec_chol
        log_det = np.log(np.diag(prec_chol)).sum()
        log_prob[:, k] = -0.5 * (n_features * np.log(2 * np.pi) + (y ** 2).sum(axis=1)) + log_det + np.log(weights[k])
    return log_prob.argmax(axis=1)

def check(name, predicted, expected):
    mismatches = int((np.asarray(predicted) != np.asarray(expected)).sum())
    print(f"{'✔' if mismatches == 0 else '✘'} {name:<38} exported parameters reproduce the labels "
          f"({len(expected) - mismatches:,}/{len(expected):,})")
    assert mismatches == 0, f"{name}: {mismatches} labels differ"

banner("Export for the web app")
X_check = rfm_clean[RFM_COLS].to_numpy() * rfm_scaler.scale_ + rfm_scaler.min_     # same arithmetic as MinMaxScaler
assert np.allclose(X_check, X_rfm)

models = {}
for name, raw, column in [("K-Means", kmeans_raw_labels, "KMeans_Cluster"),
                          ("K-Means++", kmeanspp_raw_labels, "KMeansPlusPlus_Cluster")]:
    mapping = raw_to_final(raw, rfm_clean[column])
    centers = reorder(cluster_results[name]["model"].cluster_centers_, mapping)
    check(name, nearest(X_check, centers), rfm_clean[column])
    models[name] = {"type": "centroid", "column": column, "centers_scaled": centers}

mapping = raw_to_final(kmedoids_raw_labels, rfm_clean["KMedoids_Cluster"])
medoid_rows = np.asarray(kmedoids_model.medoid_indices_)
centers = reorder({raw: X_rfm[medoid_rows[raw]] for raw in mapping}, mapping)
medoid_ids = [None] * len(mapping)
for raw, final in mapping.items():
    medoid_ids[final] = int(rfm_clean.loc[medoid_rows[raw], "CustomerID"])
check("K-Medoids", nearest(X_check, centers), rfm_clean["KMedoids_Cluster"])
models["K-Medoids"] = {"type": "centroid", "column": "KMedoids_Cluster", "centers_scaled": centers,
                       "medoid_customer_ids": medoid_ids}

mapping = raw_to_final(gmm_raw_labels, rfm_clean["GMM_Cluster"])
assert len(mapping) == gmm_model.n_components, "a GMM component has no customers"
gmm_params = {"weights": reorder({r: [w] for r, w in enumerate(gmm_model.weights_)}, mapping)[:, 0],
              "means_scaled": reorder(gmm_model.means_, mapping),
              "precisions_cholesky": np.array([gmm_model.precisions_cholesky_[raw] for raw, _ in
                                               sorted(mapping.items(), key=lambda item: item[1])])}
check("GMM", gmm_predict(X_check, gmm_params["weights"], gmm_params["means_scaled"],
                         gmm_params["precisions_cholesky"]), rfm_clean["GMM_Cluster"])
models["GMM"] = {"type": "gmm", "column": "GMM_Cluster", **gmm_params}

mapping = raw_to_final(meanshift_raw_labels, rfm_clean["MeanShift_Cluster"])
centers = reorder(meanshift_model.cluster_centers_, mapping)
check("Mean Shift", nearest(X_check, centers), rfm_clean["MeanShift_Cluster"])
models["Mean Shift"] = {"type": "centroid", "column": "MeanShift_Cluster", "centers_scaled": centers,
                        "bandwidth": meanshift_bandwidth, "quantile": meanshift_quantile}

for name in models:
    result = cluster_results[name]
    models[name].update({"n_clusters": result["Number of Clusters"], "silhouette": result["Silhouette Score"],
                         "davies_bouldin": result["Davies-Bouldin Index"], "setting": result["setting"]})

# ---- CLV models: segment model + data needed to retrain the tuned models in the app ----
clv_mapping_final = {int(raw): int(final) for raw, final in clv_mapping.items()}
clv_centers = reorder(clv_kmeans.cluster_centers_, clv_mapping_final)
X_train_scaled = clv_train[RFM_COLS].to_numpy() * clv_scaler.scale_ + clv_scaler.min_
X_test_scaled = clv_test[RFM_COLS].to_numpy() * clv_scaler.scale_ + clv_scaler.min_
check("CLV segment (training customers)", nearest(X_train_scaled, clv_centers), clv_train["Cluster"])
check("CLV segment (test customers)", nearest(X_test_scaled, clv_centers), clv_test["Cluster"])

clv_columns = ["CustomerID"] + RFM_COLS + ["Cluster", "FutureGrossRevenue", "FutureCancelledValue", "CLV_Target"]
clv_train[clv_columns].to_csv(APP_DATA / "clv_train.csv", index=False)
clv_test[clv_columns].to_csv(APP_DATA / "clv_test.csv", index=False)

meta = {
    "random_state": RANDOM_STATE,
    "features": RFM_COLS,
    "dataset": {
        "raw_rows": len(df_raw), "raw_customers": int(df_raw["CustomerID"].nunique()),
        "countries": int(df_raw["Country"].nunique()),
        "clean_rows": len(df_clean), "clean_customers": int(df_clean["CustomerID"].nunique()),
        "total_revenue": float(df_clean["Revenue"].sum()),
        "first_date": df_clean["InvoiceDate"].min(), "last_date": df_clean["InvoiceDate"].max(),
        "snapshot_date": snapshot_date,
        "rfm_customers_before_iqr": len(rfm_raw), "rfm_customers_after_iqr": len(rfm_clean),
    },
    "iqr_bounds": iqr_bounds[["Lower Bound", "Upper Bound"]].to_dict(orient="index"),
    "scaler": {"scale": rfm_scaler.scale_, "min": rfm_scaler.min_,
               "data_min": rfm_scaler.data_min_, "data_max": rfm_scaler.data_max_},
    "primary_model": PRIMARY_MODEL,
    "models": models,
    "clv": {
        "cutoff": cutoff, "observation_days": observation_days, "prediction_days": prediction_days,
        "data_start": data_start, "data_end": data_end,
        "k": int(clv_k), "feature_columns": FEATURE_COLUMNS,
        "scaler": {"scale": clv_scaler.scale_, "min": clv_scaler.min_,
                   "data_min": clv_scaler.data_min_, "data_max": clv_scaler.data_max_},
        "segment_centers_scaled": clv_centers,
        "iqr_bounds": clv_iqr_bounds[["Lower Bound", "Upper Bound"]].to_dict(orient="index"),
        "untuned_params": {"Random Forest": {"n_estimators": 100}, "Gradient Boosting": {}},
        "tuned_params": {"Random Forest": dict(rf_search.best_params_),
                         "Gradient Boosting": dict(gb_search.best_params_)},
        "best_cv_r2": {"Random Forest": rf_search.best_score_, "Gradient Boosting": gb_search.best_score_},
        "search": {"iterations": SEARCH_ITERATIONS, "cv_folds": CV_FOLDS, "test_size": TEST_SIZE},
    },
    "paper": {"clustering": PAPER_CLUSTERING.to_dict(orient="records"),
              "k_tables": {m: {str(k): list(v) for k, v in t.items()} for m, t in PAPER_K_TABLES.items()},
              "clv": PAPER_CLV.to_dict(orient="records"), "dataset": PAPER},
}
(APP_DATA / "meta.json").write_text(json.dumps(py(meta), indent=1), encoding="utf-8")

tables = {
    "customers_clustered.csv": rfm_clean,
    "rfm_raw.csv": rfm_raw,
    "preprocessing_summary.csv": preprocessing_summary,
    "monthly_revenue.csv": monthly.reset_index(),
    "elbow.csv": elbow_table,
    "k_selection_all_models.csv": pd.concat([cluster_results[m]["sweep"].assign(Model=m)
                                             for m in ["K-Means", "K-Means++", "K-Medoids", "GMM"]], ignore_index=True),
    "meanshift_bandwidth_search.csv": meanshift_search,
    "clustering_comparison.csv": clustering_comparison,
    "ari_matrix.csv": ari_matrix.reset_index().rename(columns={"index": "Model"}),
    "stability_summary.csv": stability_summary.reset_index(),
    "cluster_profiles_all_algorithms.csv": all_profiles,
    "paper_vs_ours_clustering.csv": clustering_vs_paper,
    "clv_model_results.csv": clv_model_results,
    "paper_vs_ours_clv.csv": clv_vs_paper,
    "paper_style_clv_per_cluster.csv": cluster_clv.reset_index(),
    "clv_predictions_notebook.csv": clv_predictions,
}
for file_name, table in tables.items():
    table.to_csv(APP_DATA / file_name, index=False)
print(f"\n✔ meta.json + {len(tables) + 2} CSV files written to {APP_DATA}")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", required=True, help="the project notebook (.ipynb)")
    parser.add_argument("--excel", required=True, help="Online Retail.xlsx")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"), help="output folder")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    work_dir = out_dir.parent / "_build"
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.excel, work_dir / "Online Retail.xlsx")      # the notebook finds the file in its folder

    notebook = nbformat.read(args.notebook, as_version=4)
    notebook.cells.append(nbformat.v4.new_code_cell(EXPORT_CELL.replace("__APP_DATA_DIR__", repr(str(out_dir)))))
    print(f"Running {len(notebook.cells)} cells (about 5–10 minutes) ...")
    start = time.time()
    NotebookClient(notebook, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": str(work_dir)}}).execute()
    nbformat.write(notebook, work_dir / "executed_with_export.ipynb")
    print(f"Done in {time.time() - start:.0f} s. Data written to {out_dir}")
    print("Export cell output:")
    for output in notebook.cells[-1].get("outputs", []):
        print(output.get("text", ""), end="")


if __name__ == "__main__":
    main()
