# Customer Segmentation and CLV Prediction: web app

A web front end (Streamlit) for my project notebook (code and outputs in
[`notebook/Customer_Segmentation_CLV_Code_and_Outputs.ipynb`](notebook/Customer_Segmentation_CLV_Code_and_Outputs.ipynb)).
It shows the clustering results of five algorithms (K-Means, K-Means++, K-Medoids, GMM, Mean Shift) and predicts
customer lifetime value (CLV) live with Random Forest and Gradient Boosting.

The project reproduces Ling & Weiling, "Enhancing Segmentation: A Comparative Study of Clustering Methods",
IEEE Access 13 (2025), DOI 10.1109/ACCESS.2025.3550339, on the UCI Online Retail dataset.

127018042 · Pattan Sameera Hussainy · Guide: Dr. Vidivelli S · SASTRA Deemed University

**Live app: <https://sameera-customer-segmentation-clv.streamlit.app>** (Streamlit Community Cloud; after a period without visitors the app sleeps, so the first visit may take a minute.)

## Run it

**Windows:** double-click **`run_app.bat`**. The app opens in your browser at <http://localhost:8501>.
Close the black window to stop it.

- The first start installs the libraries, which takes 1–2 minutes and needs the internet. Later starts are fast
  and work offline.
- The launcher uses `uv` if it is installed. Otherwise it creates a private Python environment in
  `%LOCALAPPDATA%\clv_segmentation_app_venv`, which needs Python 3.13 from python.org.

**Any system (manual):**

```
pip install -r requirements.txt
streamlit run app.py
```

## Pages

| Page | What it shows |
|---|---|
| Overview | Dataset size, the six pipeline steps, and monthly revenue with the CLV cutoff |
| Data & RFM | Cleaning steps (funnel), RFM features, histograms before and after outlier removal, IQR bounds, Min-Max scaling |
| Clustering results | Pick K-Means, K-Means++, K-Medoids, GMM or Mean Shift: interactive 3D clusters, cluster profiles and meaning, how K (or the bandwidth) was chosen |
| Compare algorithms | Silhouette and Davies–Bouldin for all five methods, agreement between them (ARI), the five segmentations side by side, *Reference Paper Result* vs *Our Reproduction Result*, and the K-Means++ stability test |
| Segment a customer | Type Recency, Frequency and Monetary values (or load a customer): every algorithm assigns a cluster, with GMM membership probabilities and a 3D position |
| CLV prediction | Predicts net spending in the next 100 days with tuned Random Forest and Gradient Boosting, plus MAE/MSE/RMSE/R², actual vs predicted, feature importance, best hyperparameters, and the comparison with the paper |
| Lookup & batch | Look up any customer's clusters and CLV, or upload a CSV and download the predictions (a template is provided) |

## Suggested 3-minute demo for the review

1. **Overview**: the problem, the data, and the six steps.
2. **Clustering results**, with K-Means++ selected: rotate the 3D plot and read the three segments (high-value
   loyal, recent low-spend, lapsed). Switch to K-Medoids to show the medoids are real customers.
3. **Compare algorithms**: the same metrics for all five methods, facts not a ranking, and the paper comparison.
4. **Segment a customer**: click *Loyal high spender*, then *Lapsed customer*, and watch the red diamond move.
5. **CLV prediction**: load test customer **14504**. It is predicted at about £275–£281 (actual £130.55), a customer
   the models never saw. Then open the *Model performance* tab.
6. **Lookup & batch**: score the 592 test customers and download the CSV.

## Where the numbers come from

- `data/` was produced by **`make_app_data.py`**, which runs the project notebook **unchanged** and adds one
  export cell (`python make_app_data.py --notebook notebook/Customer_Segmentation_CLV_Code_and_Outputs.ipynb
  --excel "Online Retail.xlsx"`). That cell saves the fitted model parameters, every customer's cluster labels and the result tables.
- **Clustering** uses the exported parameters directly: centroids, medoids, GMM components and Mean Shift centres.
  Nothing is refitted, so every customer keeps exactly the notebook's cluster.
- **CLV models** are retrained when the app starts, with the notebook's tuned hyperparameters on the notebook's own
  2,365 training customers. This gives the notebook's predictions exactly.
- **`tests/test_core.py`** checks all of this (7 checks):

  ```
  python tests/test_core.py
  ```

  It confirms that all 3,710 customers keep their cluster in all five algorithms, that the CLV predictions and the
  MAE/MSE/RMSE/R² values equal the notebook's, and that uploaded files are validated.

## Files

```
notebook/              the project notebook (code and outputs)
app.py                 the web pages (Streamlit)
clv_core.py            prediction logic: clustering assignment, CLV models, validation (no UI code)
make_app_data.py       rebuilds data/ from the notebook (optional; needs requirements-build.txt)
data/                  exported parameters (meta.json) and result tables (CSV)
assets/sastra_logo.png sidebar logo
tests/test_core.py     reproduction checks
run_app.bat            Windows launcher
requirements.txt       exact library versions
.streamlit/config.toml theme and settings (no usage statistics are sent)
```

## Honest notes

- The CLV target is **future** net spending after 1 Sep 2011, predicted only from behaviour before that date, so
  no future information leaks into the inputs. R² is about 0.15–0.18. The paper reports R² ≈ 0.9999 without
  defining its target; notebook Section 26.4 shows that an in-period target produces such values.
- Segment names (for example "High-value loyal") and the suggested actions are rules of thumb. They come from each
  cluster's R/F/M level compared with the overall average, and are meant to make the clusters easy to read.

## Data and credits

- Dataset: Online Retail, UCI Machine Learning Repository (D. Chen, 2015), CC BY 4.0.
  <https://doi.org/10.24432/C5BW33>. Only customer-level aggregates (RFM) and model results are stored in `data/`.
- Base paper: L. S. Ling and C. T. Weiling, IEEE Access, 2025.
