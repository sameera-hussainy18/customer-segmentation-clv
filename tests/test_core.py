"""Checks that the app reproduces the notebook exactly.

Run from the app folder:   python tests/test_core.py      (or: python -m pytest tests)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clv_core as core  # noqa: E402

META = core.load_meta()
CUSTOMERS = core.load_table("customers_clustered.csv")


def test_every_customer_keeps_the_notebook_cluster():
    segmenter = core.Segmenter(META)
    for name in core.MODEL_ORDER:
        column = META["models"][name]["column"]
        predicted = segmenter.predict(name, CUSTOMERS)
        assert np.array_equal(predicted, CUSTOMERS[column].to_numpy()), name


def test_gmm_probabilities_are_valid():
    prob = core.Segmenter(META).gmm_probabilities(CUSTOMERS.head(50))
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert np.array_equal(prob.argmax(axis=1), CUSTOMERS["GMM_Cluster"].head(50).to_numpy())


def test_medoids_are_the_notebook_customers():
    centers = core.Segmenter(META).centers("K-Medoids")
    assert centers["Medoid customer"].tolist() == [14188, 15068, 16823]
    medoid = CUSTOMERS.set_index("CustomerID").loc[14188, core.RFM].to_numpy(dtype=float)
    assert np.allclose(centers.loc[0, core.RFM].to_numpy(dtype=float), medoid)


def test_clv_models_reproduce_the_notebook():
    train, test = core.load_table("clv_train.csv"), core.load_table("clv_test.csv")
    clv = core.CLVPredictor(META, train, test).fit()
    ours = clv.test_predictions()
    notebook = core.load_table("clv_predictions_notebook.csv")
    assert ours["CustomerID"].tolist() == notebook["CustomerID"].tolist()
    for name, variant in core.CLV_MODELS:
        assert np.allclose(ours[f"{name} ({variant})"], notebook[f"Predicted: {name} ({variant})"], rtol=0, atol=1e-6)
    metrics = clv.evaluate().set_index(["Model", "Tuned/Untuned"])
    reported = core.load_table("clv_model_results.csv").set_index(["Model", "Tuned/Untuned"])
    assert np.allclose(metrics[["MAE", "MSE", "RMSE", "R²"]], reported.loc[metrics.index, ["MAE", "MSE", "RMSE", "R²"]])
    # the segment feature is recomputed from the exported model for test customers
    assert np.array_equal(clv.segment(test), test["Cluster"].to_numpy())
    # one new customer: a single prediction per model, segment within range
    single = clv.predict([[30, 3, 900.0]])
    assert len(single) == 1 and 0 <= single.loc[0, "Segment"] < clv.k


def test_descriptions_match_the_notebook_profile():
    profile = core.profile_clusters(CUSTOMERS, CUSTOMERS["KMeansPlusPlus_Cluster"])
    described = core.describe_clusters(profile, CUSTOMERS)
    assert profile["Customers"].tolist() == [828, 1964, 918]
    assert described.loc[0, "Description"] == "recently active, frequent and high-spending purchasing behaviour"
    assert described.loc[2, "Description"] == "less recent and lower-engagement purchasing behaviour"


def test_segment_names_agree_with_descriptions():
    names = {}
    for name in core.MODEL_ORDER:
        column = META["models"][name]["column"]
        profile = core.profile_clusters(CUSTOMERS, CUSTOMERS[column])
        names[name] = core.describe_clusters(profile, CUSTOMERS)["Segment"].tolist()
    assert names["K-Means++"] == ["High-value loyal", "Recent, low spend", "Lapsed"]
    assert names["Mean Shift"] == ["Active, average value", "Lapsed"]
    assert all(segment in core.ACTIONS for segments in names.values() for segment in segments)


def test_uploaded_tables_are_validated():
    good = pd.DataFrame({"customerid": [1, 2, 3, 4, 5, 6], "RECENCY": [10, -1, 5, 7, "x", 3],
                         "Frequency": [2, 2, 0, 1, 1, 4], " monetary ": [100.0, 50.0, 20.0, float("inf"), 9.0, 80.0]})
    cleaned, skipped = core.clean_rfm_table(good)
    assert cleaned["CustomerID"].tolist() == ["1", "6"] and skipped == 4
    for bad in [pd.DataFrame({"Recency": [1], "Frequency": [1]}),                  # no Monetary column
                pd.DataFrame({"Recency": ["a"], "Frequency": [1], "Monetary": [5.0]})]:   # nothing valid
        try:
            core.clean_rfm_table(bad)
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")


if __name__ == "__main__":
    tests = [value for key, value in dict(globals()).items() if key.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
