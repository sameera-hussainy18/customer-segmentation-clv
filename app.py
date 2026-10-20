"""Customer Segmentation and CLV Prediction: a web front end for the project notebook.

Start it with   streamlit run app.py      (on Windows you can double-click run_app.bat)
Every number comes from the project notebook (see clv_core.py and make_app_data.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import clv_core as core

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title="Customer Segmentation & CLV", page_icon=":material/insights:", layout="wide")

RED, INDIGO, GRAY = "#C00000", "#3C3C8C", "#6B7280"
CLUSTER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
TEXT_COLORS = ["blue", "orange", "green", "red", "violet", "gray", "gray", "gray"]   # same order, for st.markdown
FEATURE_COLORS = {"Recency": "#1f77b4", "Frequency": "#2ca02c", "Monetary": "#d62728"}
UNITS = {"Recency": "days since last purchase", "Frequency": "orders", "Monetary": "£ spent"}
CENTRE_NAMES = {"K-Means": "Centroids", "K-Means++": "Centroids", "K-Medoids": "Medoids (real customers)",
                "GMM": "Component means", "Mean Shift": "Density peaks"}


# ------------------------------------------------------------------------------------------- cached data
@st.cache_data
def meta() -> dict:
    return core.load_meta()


@st.cache_data
def table(name: str) -> pd.DataFrame:
    return core.load_table(name)


@st.cache_data
def customers() -> pd.DataFrame:
    return table("customers_clustered.csv")


@st.cache_resource
def segmenter() -> core.Segmenter:
    return core.Segmenter(meta())


@st.cache_resource(show_spinner="Training the CLV models with the notebook's tuned settings (first start only) ...")
def clv_models() -> core.CLVPredictor:
    return core.CLVPredictor(meta(), table("clv_train.csv"), table("clv_test.csv")).fit()


@st.cache_data
def cluster_descriptions(model_name: str) -> pd.DataFrame:
    data = customers()
    profile = core.profile_clusters(data, data[meta()["models"][model_name]["column"]])
    return profile.merge(core.describe_clusters(profile, data), on="Cluster")


@st.cache_data
def clv_segment_descriptions() -> pd.DataFrame:
    train = table("clv_train.csv")
    profile = core.profile_clusters(train, train["Cluster"])
    return profile.merge(core.describe_clusters(profile, train), on="Cluster")


# ------------------------------------------------------------------------------------------------ helpers
def show(fig: go.Figure, key: str | None = None) -> None:
    """Draw a Plotly chart; the mouse wheel scrolls the page instead of zooming the chart."""
    st.plotly_chart(fig, key=key, config={"scrollZoom": False, "displaylogo": False})


def money(value: float) -> str:
    return f"£{value:,.2f}"


def styled(fig: go.Figure, height: int, legend: bool = True) -> go.Figure:
    """Common layout: legend along the top edge, above any subplot titles (chart titles are st.markdown)."""
    top = 25 + (30 if legend else 0) + (25 if fig.layout.annotations else 0)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), showlegend=legend,
                      legend=dict(orientation="h", yref="container", y=1.0, yanchor="top", xanchor="left", x=0,
                                  itemsizing="constant"))
    return fig


def cluster_tag(cluster, name: str) -> str:
    cluster = int(cluster)
    return f":{TEXT_COLORS[cluster % len(TEXT_COLORS)]}[**Cluster {cluster} · {name}**]"


def init_inputs(prefix: str, values: tuple) -> None:
    for key, value in zip(("r", "f", "m"), values):
        st.session_state.setdefault(f"{prefix}_{key}", value)


def set_inputs(prefix: str, values: tuple, loaded_id=None) -> None:
    """Fill the three RFM inputs and remember which customer (if any) they came from."""
    r, f, m = values
    st.session_state[f"{prefix}_r"], st.session_state[f"{prefix}_f"] = int(r), int(f)
    st.session_state[f"{prefix}_m"] = float(m)
    st.session_state[f"{prefix}_loaded"] = loaded_id


def apply_example(prefix: str, values: tuple) -> None:
    """Callback of the example buttons: fill the inputs and clear the customer picker."""
    set_inputs(prefix, values)
    st.session_state[f"{prefix}_pick"] = None


def load_customer(prefix: str, source: str) -> None:
    """Callback of the customer pickers: copy the chosen customer's R, F, M into the inputs."""
    cid = st.session_state.get(f"{prefix}_pick")
    if cid is None:
        return
    frame = customers() if source == "customers" else clv_models().test
    row = frame.set_index("CustomerID").loc[cid]
    set_inputs(prefix, (row["Recency"], row["Frequency"], row["Monetary"]), loaded_id=cid)


def rfm_inputs(prefix: str, recency_label: str, recency_max: int) -> tuple[int, int, float]:
    cols = st.columns(3)
    r = cols[0].number_input(recency_label, min_value=0, max_value=recency_max, step=1, key=f"{prefix}_r")
    f = cols[1].number_input("Frequency (number of orders)", min_value=1, max_value=1000, step=1, key=f"{prefix}_f")
    m = cols[2].number_input("Monetary (total spend, £)", min_value=0.0, max_value=1_000_000.0, step=50.0,
                             format="%.2f", key=f"{prefix}_m")
    return int(r), int(f), float(m)


def example_buttons(prefix: str, examples: dict) -> None:
    row = st.container(horizontal=True)
    row.caption("Examples:")
    for label, values in examples.items():
        row.button(label, key=f"{prefix}_ex_{label}", on_click=apply_example, args=(prefix, values), type="tertiary",
                   icon=":material/person:")


# ------------------------------------------------------------------------------------------------ figures
def cluster_3d(data: pd.DataFrame, labels, centers: pd.DataFrame | None = None, centre_name: str = "Centres",
               highlight: tuple | None = None, height: int = 560) -> go.Figure:
    labels = np.asarray(labels)
    fig = go.Figure()
    for c in sorted(np.unique(labels)):
        part = data[labels == c]
        fig.add_trace(go.Scatter3d(
            x=part["Recency"], y=part["Frequency"], z=part["Monetary"], mode="markers", name=f"Cluster {c}",
            marker=dict(size=2.6, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], opacity=0.7),
            customdata=part[["CustomerID"]],
            hovertemplate="Customer %{customdata[0]}<br>Recency %{x} days<br>Frequency %{y}<br>"
                          "Monetary £%{z:,.2f}<extra>Cluster " + str(c) + "</extra>"))
    if centers is not None:
        fig.add_trace(go.Scatter3d(
            x=centers["Recency"], y=centers["Frequency"], z=centers["Monetary"], mode="markers+text",
            name=centre_name, text=[f"C{c}" for c in centers["Cluster"]], textposition="top center",
            marker=dict(size=6, color="black", symbol="diamond"),
            hovertemplate="%{text}: R %{x:.0f} · F %{y:.1f} · M £%{z:,.0f}<extra>" + centre_name + "</extra>"))
    if highlight is not None:
        fig.add_trace(go.Scatter3d(
            x=[highlight[0]], y=[highlight[1]], z=[highlight[2]], mode="markers", name="This customer",
            marker=dict(size=11, color=RED, symbol="diamond", line=dict(color="white", width=2)),
            hovertemplate="This customer<br>R %{x} · F %{y} · M £%{z:,.2f}<extra></extra>"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, itemsizing="constant"),
                      scene=dict(xaxis_title="Recency (days)", yaxis_title="Frequency (orders)",
                                 zaxis_title="Monetary (£)", camera=dict(eye=dict(x=1.55, y=1.55, z=0.85))))
    return fig


def cluster_2d(data: pd.DataFrame, labels, height: int = 380) -> go.Figure:
    labels = np.asarray(labels)
    jitter = data["Frequency"].to_numpy() + np.random.default_rng(42).uniform(-0.25, 0.25, len(data))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Recency vs Monetary", "Frequency (jittered) vs Monetary"])
    for c in sorted(np.unique(labels)):
        mask = labels == c
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        fig.add_trace(go.Scattergl(x=data.loc[mask, "Recency"], y=data.loc[mask, "Monetary"], mode="markers",
                                   marker=dict(size=4, color=color, opacity=0.55), name=f"Cluster {c}",
                                   legendgroup=str(c)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=jitter[mask], y=data.loc[mask, "Monetary"], mode="markers",
                                   marker=dict(size=4, color=color, opacity=0.55), name=f"Cluster {c}",
                                   legendgroup=str(c), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="Recency (days)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (orders)", row=1, col=2)
    fig.update_yaxes(title_text="Monetary (£)", row=1, col=1)
    return styled(fig, height)


def k_selection_figure(name: str) -> go.Figure:
    m = meta()
    if name == "Mean Shift":
        search = table("meanshift_bandwidth_search.csv")
        chosen = m["models"]["Mean Shift"]["bandwidth"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Clusters found vs bandwidth",
                                                            "Scores vs bandwidth (one cluster: no score)"])
        fig.add_trace(go.Scatter(x=search["Bandwidth"], y=search["Number of Clusters"], mode="lines+markers",
                                 name="Number of clusters", line=dict(color="#9467bd")), row=1, col=1)
        fig.add_trace(go.Scatter(x=search["Bandwidth"], y=search["Silhouette Score"], mode="lines+markers",
                                 name="Silhouette ↑", line=dict(color="#2ca02c")), row=1, col=2)
        fig.add_trace(go.Scatter(x=search["Bandwidth"], y=search["Davies-Bouldin Index"], mode="lines+markers",
                                 name="Davies–Bouldin ↓", line=dict(color="#d62728")), row=1, col=2)
        for col in (1, 2):
            fig.add_vline(x=chosen, line_dash="dot", line_color="black", row=1, col=col)
            fig.update_xaxes(title_text="Bandwidth", row=1, col=col)
        return styled(fig, 340)

    sweep = table("k_selection_all_models.csv")
    sweep = sweep[sweep["Model"] == name]
    paper = m["paper"]["k_tables"].get(name, {})
    chosen = m["models"][name]["n_clusters"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette Score (higher = better)",
                                                        "Davies–Bouldin Index (lower = better)"])
    for col, (column, color) in enumerate([("Silhouette Score", "#2ca02c"), ("Davies-Bouldin Index", "#d62728")], 1):
        fig.add_trace(go.Scatter(x=sweep["K"], y=sweep[column], mode="lines+markers", line=dict(color=color),
                                 name="Our Reproduction Result", legendgroup="ours", showlegend=col == 1),
                      row=1, col=col)
        if paper:
            ks = sorted(int(k) for k in paper)
            fig.add_trace(go.Scatter(x=ks, y=[paper[str(k)][col - 1] for k in ks], mode="lines+markers",
                                     line=dict(color="gray", dash="dash"), marker=dict(symbol="square"),
                                     name="Reference Paper Result", legendgroup="paper", showlegend=col == 1),
                          row=1, col=col)
        fig.add_vline(x=chosen, line_dash="dot", line_color="black", row=1, col=col)
        fig.update_xaxes(title_text="Number of clusters K", dtick=1, row=1, col=col)
    return styled(fig, 340)


def monthly_revenue_figure() -> go.Figure:
    data = table("monthly_revenue.csv")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    cutoff = pd.Timestamp(meta()["clv"]["cutoff"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=data["InvoiceDate"], y=data["Revenue"], name="Revenue (£)", marker_color="#1f77b4",
                         hovertemplate="%{x|%b %Y}: £%{y:,.0f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=data["InvoiceDate"], y=data["ActiveCustomers"], name="Active customers",
                             mode="lines+markers", line=dict(color="#ff7f0e"),
                             hovertemplate="%{x|%b %Y}: %{y:,} customers<extra></extra>"), secondary_y=True)
    fig.add_shape(type="line", x0=cutoff, x1=cutoff, y0=0, y1=1, yref="paper", line=dict(color=RED, dash="dash"))
    fig.add_annotation(x=cutoff, y=1.0, yref="paper", text=" CLV cutoff", showarrow=False, xanchor="left",
                       yanchor="top", font=dict(color=RED))
    fig.update_yaxes(title_text="Revenue (£)", tickprefix="£", tickformat=",.0f", secondary_y=False)
    fig.update_yaxes(title_text="Active customers", showgrid=False, rangemode="tozero", secondary_y=True)
    return styled(fig, 360)


# ================================================================================================== pages
def page_overview():
    ds = meta()["dataset"]
    st.title("Customer Segmentation and Customer Lifetime Value Prediction")
    st.markdown(
        "Groups online-retail customers by **Recency, Frequency and Monetary (RFM)** behaviour with **five "
        "clustering methods**, then predicts each customer's **future spending (CLV)** with **Random Forest** and "
        "**Gradient Boosting**. It reproduces *Ling & Weiling, “Enhancing Segmentation: A Comparative Study of "
        "Clustering Methods”, IEEE Access 2025* on the UCI Online Retail dataset.")
    first, last = pd.Timestamp(ds["first_date"]), pd.Timestamp(ds["last_date"])
    cards = st.columns(5)
    cards[0].metric("Transactions", f"{ds['raw_rows']:,}", border=True, help="Rows in the raw Excel file")
    cards[1].metric("After cleaning", f"{ds['clean_rows']:,}", border=True)
    cards[2].metric("Customers", f"{ds['rfm_customers_after_iqr']:,}", border=True,
                    help=f"Customers clustered: {ds['rfm_customers_before_iqr']:,} before IQR outlier removal")
    cards[3].metric("Revenue", f"£{ds['total_revenue'] / 1e6:.2f} M", border=True, help="Revenue of the cleaned data")
    cards[4].metric("Period", f"{(last - first).days} days", border=True,
                    help=f"{first:%d %b %Y} to {last:%d %b %Y}, customers in {ds['countries']} countries")

    st.subheader("How the project works")
    steps = [("Online Retail data", f"{ds['raw_rows']:,} transactions"),
             ("Cleaning", f"{ds['clean_rows']:,} transactions"),
             ("RFM per customer", f"{ds['rfm_customers_before_iqr']:,} customers"),
             ("IQR outliers + Min-Max", f"{ds['rfm_customers_after_iqr']:,} customers"),
             ("Five clustering methods", "K-Means · K-Means++ · K-Medoids · GMM · Mean Shift"),
             ("CLV prediction", "Random Forest · Gradient Boosting")]
    for i, (col, (title, detail)) in enumerate(zip(st.columns(len(steps)), steps), 1):
        with col.container(border=True, height="stretch"):
            st.markdown(f":red[**{i}**] **{title}**" + (" →" if i < len(steps) else ""))
            st.caption(detail)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Monthly revenue and active customers")
        show(monthly_revenue_figure())
        st.caption("December 2011 contains only 9 days of data. Features for CLV come from before the cutoff, "
                   "the target from after it.")
    with right:
        st.subheader("What each page shows")
        st.markdown(
            "- **Data & RFM**: cleaning steps, RFM features, outlier removal and scaling\n"
            "- **Clustering results**: explore each algorithm in 3D, its cluster profiles and how K was chosen\n"
            "- **Compare algorithms**: Silhouette, Davies–Bouldin, agreement (ARI) and the paper's results\n"
            "- **Segment a customer**: type R, F, M and see the cluster every algorithm assigns\n"
            "- **CLV prediction**: predict a customer's next-100-day spend; model accuracy and tuning\n"
            "- **Lookup & batch**: any customer's results, or predictions for a whole CSV file")
        st.info("Every number here comes from the project notebook: the app uses the parameters the notebook "
                "fitted, and its checks confirm identical clusters and CLV predictions.", icon=":material/verified:")
    clv_models()          # warm-up: train the CLV models now, so the prediction pages open instantly later


def page_data():
    m = meta()
    ds = m["dataset"]
    st.title("Data preparation and RFM features")
    summary = table("preprocessing_summary.csv")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Cleaning steps")
        fig = go.Figure(go.Funnel(y=summary["Step"], x=summary["Rows Remaining"], textinfo="value+percent initial",
                                  marker=dict(color=[INDIGO] + ["#7B7BB8"] * (len(summary) - 2) + [RED])))
        show(styled(fig, 380, legend=False))
    with right:
        st.subheader("Rows and customers after each step")
        st.dataframe(summary[["Step", "Rows Remaining", "Rows Removed", "Customers"]], hide_index=True, height=290,
                     column_config={"Step": st.column_config.TextColumn(width="medium"),
                                    "Rows Remaining": st.column_config.NumberColumn("Rows left", format="localized"),
                                    "Rows Removed": st.column_config.NumberColumn("Removed", format="localized"),
                                    "Customers": st.column_config.NumberColumn(format="localized")})
        st.caption(f"Revenue = Quantity × UnitPrice → total £{ds['total_revenue']:,.2f}")

    st.subheader("RFM: three numbers per customer")
    cards = st.columns(3)
    texts = {"Recency": "Days since the customer's **last purchase** (snapshot = last invoice date + 1 day). "
                        "**Low = recent.**",
             "Frequency": "Number of **different invoices** (orders) the customer placed.",
             "Monetary": "**Total revenue** from the customer (£)."}
    for card, (feature, text) in zip(cards, texts.items()):
        with card.container(border=True):
            st.markdown(f"**{feature} ({feature[0]})**")
            st.markdown(text)

    view = st.segmented_control("Customers shown", ["Before outlier removal", "After outlier removal"],
                                default="After outlier removal", required=True, key="hist_view")
    raw = table("rfm_raw.csv")
    data = raw if view.startswith("Before") else customers()
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{f} ({UNITS[f]})" for f in core.RFM])
    for i, feature in enumerate(core.RFM, 1):
        fig.add_trace(go.Histogram(x=data[feature], nbinsx=50, marker_color=FEATURE_COLORS[feature], name=feature),
                      row=1, col=i)
    show(styled(fig, 320, legend=False))
    st.caption(f"{len(data):,} customers. Maximum Frequency {data['Frequency'].max():,} orders, maximum Monetary "
               f"£{data['Monetary'].max():,.2f}: Frequency and Monetary are strongly right-skewed before outliers "
               "are removed.")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Outlier removal (IQR rule)")
        rows = []
        for feature, bounds in m["iqr_bounds"].items():
            outside = ((raw[feature] < bounds["Lower Bound"]) | (raw[feature] > bounds["Upper Bound"])).sum()
            rows.append({"Feature": feature, "Lower bound": bounds["Lower Bound"],
                         "Upper bound": bounds["Upper Bound"], "Customers outside": int(outside)})
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"Lower bound": st.column_config.NumberColumn(format="%.2f"),
                                    "Upper bound": st.column_config.NumberColumn(format="%.2f")})
        before, after = ds["rfm_customers_before_iqr"], ds["rfm_customers_after_iqr"]
        st.caption(f"A customer is removed if any feature is outside Q1 − 1.5·IQR … Q3 + 1.5·IQR: "
                   f"{before:,} → {after:,} customers ({(before - after) / before:.1%} removed).")
    with right:
        st.subheader("Min-Max scaling")
        st.latex(r"x' = \frac{x - \min(x)}{\max(x) - \min(x)} \in [0, 1]")
        scaler = m["scaler"]
        st.dataframe(pd.DataFrame({"Feature": core.RFM, "Minimum": scaler["data_min"], "Maximum": scaler["data_max"]}),
                     hide_index=True, column_config={"Minimum": st.column_config.NumberColumn(format="%.2f"),
                                                     "Maximum": st.column_config.NumberColumn(format="%.2f")})
        st.caption("After scaling, days, orders and pounds weigh equally in the distances used by clustering.")

    with st.expander(f"All {len(customers()):,} segmented customers"):
        st.dataframe(customers()[["CustomerID"] + core.RFM], hide_index=True, height=360,
                     column_config={"CustomerID": st.column_config.NumberColumn(format="%d"),
                                    "Monetary": st.column_config.NumberColumn("Monetary (£)", format="localized")})


def page_clustering():
    m = meta()
    st.title("Clustering results")
    name = st.segmented_control("Algorithm", core.MODEL_ORDER, default=m["primary_model"], required=True,
                                key="explore_model")
    info = m["models"][name]
    data = customers()
    labels = data[info["column"]].to_numpy()
    profile = cluster_descriptions(name)

    cards = st.columns(4)
    cards[0].metric("Clusters", info["n_clusters"], border=True)
    cards[1].metric("Silhouette Score ↑", f"{info['silhouette']:.4f}", border=True,
                    help="−1 to 1: how much closer each customer is to its own cluster than to the nearest other one")
    cards[2].metric("Davies–Bouldin Index ↓", f"{info['davies_bouldin']:.4f}", border=True,
                    help="≥ 0: average overlap between each cluster and its most similar cluster")
    cards[3].metric("Largest cluster", f"{profile['% of Customers'].max():.1f} %", border=True,
                    help="Share of the customers in the biggest cluster")
    st.caption(f"Setting: {info['setting']} · random_state = {m['random_state']}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        show(cluster_3d(data, labels, segmenter().centers(name), CENTRE_NAMES[name]), key=f"cluster3d_{name}")
        st.caption("Drag to rotate, double-click to reset. Black diamonds: "
                   f"{CENTRE_NAMES[name].lower()}.")
    with right:
        st.subheader("What the clusters mean")
        for _, row in profile.iterrows():
            st.markdown(f"{cluster_tag(row['Cluster'], row['Segment'])}: {row['Description']}.  \n"
                        f"*{row['Suggested action']}.*")
        st.caption("Segment names and actions are rules of thumb from each cluster's R/F/M level compared with "
                   "the overall average (Recency: low = recent).")
        if name == "K-Medoids":
            st.markdown("**The medoids are real customers**")
            st.dataframe(segmenter().centers(name), hide_index=True,
                         column_config={"Recency": st.column_config.NumberColumn(format="%.0f"),
                                        "Frequency": st.column_config.NumberColumn(format="%.0f"),
                                        "Monetary": st.column_config.NumberColumn("Monetary (£)", format="%.2f")})

    st.subheader("Cluster profiles")
    st.dataframe(profile[["Cluster", "Segment", "Customers", "% of Customers", "Mean Recency", "Mean Frequency",
                          "Mean Monetary", "Total Revenue", "% of Revenue"]].round({"Total Revenue": 2}), hide_index=True,
                 column_config={"Segment": st.column_config.TextColumn(width="medium"),
                                "% of Customers": st.column_config.NumberColumn(format="%.1f %%"),
                                "Mean Recency": st.column_config.NumberColumn("Mean Recency (days)", format="%.1f"),
                                "Mean Frequency": st.column_config.NumberColumn(format="%.2f"),
                                "Mean Monetary": st.column_config.NumberColumn("Mean Monetary (£)", format="%.2f"),
                                "Total Revenue": st.column_config.NumberColumn("Total Revenue (£)", format="localized"),
                                "% of Revenue": st.column_config.NumberColumn(format="%.1f %%")})

    st.subheader("How the number of clusters was chosen" if name != "Mean Shift" else "How the bandwidth was chosen")
    show(k_selection_figure(name), key=f"ksel_{name}")
    if name == "Mean Shift":
        st.caption("Mean Shift finds the number of clusters itself; the bandwidth (from estimate_bandwidth "
                   "quantiles 0.05–0.30) controls it. The setting with the highest Silhouette was kept.")
    else:
        st.caption("Rule used in the paper: highest Silhouette Score, with the Davies–Bouldin Index as a check. "
                   "The paper's tables only start at K = 4 (K-Means++), 6 (K-Medoids) or 8 (GMM).")
    st.subheader("Two-dimensional views")
    show(cluster_2d(data, labels), key=f"cluster2d_{name}")


def page_compare():
    m = meta()
    st.title("Comparing the five clustering algorithms")
    comparison = table("clustering_comparison.csv")
    st.dataframe(comparison.drop(columns="Number of Customers"), hide_index=True,
                 column_config={"Number of Clusters": st.column_config.NumberColumn("Clusters"),
                                "Silhouette Score": st.column_config.NumberColumn("Silhouette ↑", format="%.4f"),
                                "Davies-Bouldin Index": st.column_config.NumberColumn("Davies–Bouldin ↓",
                                                                                      format="%.4f"),
                                "Smallest Cluster": st.column_config.NumberColumn("Smallest cluster"),
                                "Largest Cluster (%)": st.column_config.NumberColumn("Largest cluster",
                                                                                     format="%.1f %%"),
                                "Setting": st.column_config.TextColumn(width="large")})
    st.caption(f"All five algorithms cluster the same {len(customers()):,} customers on the same scaled R, F, M.")
    ari = table("ari_matrix.csv").set_index("Model")
    c = comparison.set_index("Model")
    st.info(
        f"**Facts from the metrics, not a ranking.** K-Means and K-Means++ give identical segments "
        f"(ARI = {ari.loc['K-Means', 'K-Means++']:.2f}) with Silhouette {c.loc['K-Means++', 'Silhouette Score']:.4f} "
        f"and DBI {c.loc['K-Means++', 'Davies-Bouldin Index']:.4f}. K-Medoids is almost the same "
        f"(ARI = {ari.loc['K-Means++', 'K-Medoids']:.3f}). GMM has the lowest Silhouette and highest DBI. Mean Shift "
        f"finds {int(c.loc['Mean Shift', 'Number of Clusters'])} clusters, the larger holding "
        f"{c.loc['Mean Shift', 'Largest Cluster (%)']:.1f} % of customers.", icon=":material/balance:")

    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Silhouette Score (higher = better)",
                                                            "Davies–Bouldin Index (lower = better)"])
        for col, (column, color) in enumerate([("Silhouette Score", "#4A6FA5"), ("Davies-Bouldin Index", RED)], 1):
            fig.add_trace(go.Bar(x=comparison["Model"], y=comparison[column], marker_color=color,
                                 text=comparison[column].round(4), textposition="outside", name=column),
                          row=1, col=col)
        fig.update_yaxes(range=[0, 1.4], row=1, col=2)
        fig.update_yaxes(range=[0, 0.6], row=1, col=1)
        show(styled(fig, 380, legend=False))
    with right:
        fig = go.Figure(go.Heatmap(z=ari.to_numpy(), x=ari.columns, y=ari.index, colorscale="Blues", zmin=0, zmax=1,
                                   text=np.round(ari.to_numpy(), 2), texttemplate="%{text:.2f}",
                                   colorbar=dict(title="ARI")))
        st.markdown("**Agreement between segmentations** (ARI: 1 = identical, 0 = chance)")
        fig.update_yaxes(autorange="reversed")
        show(styled(fig, 380, legend=False))

    st.subheader("All five segmentations side by side")
    data = customers()
    fig = make_subplots(rows=1, cols=5, shared_yaxes=True,
                        subplot_titles=[f"{n} (K = {m['models'][n]['n_clusters']})" for n in core.MODEL_ORDER])
    in_legend = set()
    for col, name in enumerate(core.MODEL_ORDER, 1):
        labels = data[m["models"][name]["column"]].to_numpy()
        for cl in sorted(np.unique(labels)):
            mask = labels == cl
            fig.add_trace(go.Scattergl(x=data.loc[mask, "Recency"], y=data.loc[mask, "Monetary"], mode="markers",
                                       marker=dict(size=3, color=CLUSTER_COLORS[cl], opacity=0.5),
                                       name=f"Cluster {cl}", legendgroup=str(cl), showlegend=cl not in in_legend),
                          row=1, col=col)
            in_legend.add(cl)
        fig.update_xaxes(title_text="Recency (days)", row=1, col=col)
    fig.update_yaxes(title_text="Monetary (£)", row=1, col=1)
    show(styled(fig, 340))

    st.subheader("Reference Paper Result vs Our Reproduction Result")
    paper = table("paper_vs_ours_clustering.csv")
    shown = pd.DataFrame({
        "Algorithm": paper["Model"],
        "Paper's K range": paper["Paper K range tested"],
        "Reference Paper Result": [f"K = {k} · {s:.4f} / {d:.4f}" for k, s, d in
                                   zip(paper["Paper K"], paper["Paper Silhouette"], paper["Paper Davies-Bouldin"])],
        "Our Reproduction Result": [f"K = {k} · {s:.4f} / {d:.4f}" for k, s, d in
                                    zip(paper["Our K"], paper["Our Silhouette"], paper["Our Davies-Bouldin"])],
        "Ours at the paper's K": [f"K = {k} · {s:.4f} / {d:.4f}" for k, s, d in
                                  zip(paper["Our K nearest paper K"], paper["Our Silhouette at paper K"],
                                      paper["Our Davies-Bouldin at paper K"])],
    })
    st.dataframe(shown, hide_index=True)
    st.caption("Values are Silhouette / Davies–Bouldin. We searched K = 2–10 (Mean Shift: 6 bandwidths). The paper "
               "used a larger 2009–2011 version of the data (1,067,372 rows) and its K tables start at 4, 6 or 8, "
               "so exact values are not expected to match; the patterns do.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Silhouette Score: paper vs ours**")
        fig = go.Figure()
        for label, column, color in [("Reference Paper Result", "Paper Silhouette", "#8C8C8C"),
                                     ("Our Reproduction Result (our K)", "Our Silhouette", RED),
                                     ("Our Reproduction Result (paper's K)", "Our Silhouette at paper K", "#4A6FA5")]:
            fig.add_trace(go.Bar(x=paper["Model"], y=paper[column], name=label, marker_color=color,
                                 text=paper[column].round(2), textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, 0.6]), barmode="group")
        show(styled(fig, 380))
    with right:
        st.markdown("**Does K-Means++ improve stability?**")
        stability = table("stability_summary.csv")
        fig = go.Figure()
        for init, color in [("random", "#A9B7D6"), ("k-means++", RED)]:
            part = stability[stability["Initialisation"] == init]
            fig.add_trace(go.Bar(x=[f"K = {k}" for k in part["K"]], y=part["Runs_Reaching_Best"],
                                 name="Random initialisation" if init == "random" else "K-Means++",
                                 marker_color=color, text=part["Runs_Reaching_Best"], textposition="outside"))
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 22], title="Runs reaching the best solution (of 20)"))
        show(styled(fig, 340))
        st.caption("20 single-start runs per method (seeds 42–61). At the paper's K = 4, k-means++ starts reach "
                   "the best solution far more often, as the paper claims.")


def page_segment():
    m = meta()
    primary = m["primary_model"]
    data = customers()
    st.title("Segment a customer")
    st.markdown("Type a customer's **Recency, Frequency and Monetary** values. Every algorithm assigns the customer "
                "with the parameters fitted in the notebook (nothing is refitted).")
    medians = data[core.RFM].median()
    init_inputs("seg", (int(medians["Recency"]), int(medians["Frequency"]), round(float(medians["Monetary"]), 2)))
    example_buttons("seg", {"Loyal high spender": (12, 8, 2500.0), "Recent, low spend": (30, 1, 250.0),
                            "Lapsed customer": (250, 1, 300.0)})
    st.selectbox("…or load a customer from the data", data["CustomerID"].tolist(), index=None, key="seg_pick",
                 placeholder="Type or choose a customer ID", on_change=load_customer, args=("seg", "customers"))
    r, f, mo = rfm_inputs("seg", "Recency (days since last purchase)", 1000)

    predictions = segmenter().predict_all([[r, f, mo]]).iloc[0]
    rows = []
    for name in core.MODEL_ORDER:
        cluster = int(predictions[name])
        d = cluster_descriptions(name).set_index("Cluster").loc[cluster]
        rows.append({"Algorithm": name, "Cluster": cluster, "Segment": d["Segment"],
                     "Cluster size": int(d["Customers"]), "Description": d["Description"]})
    results = pd.DataFrame(rows)

    left, right = st.columns([2, 3], gap="large")
    with left:
        top = results.set_index("Algorithm").loc[primary]
        d = cluster_descriptions(primary).set_index("Cluster").loc[int(top["Cluster"])]
        with st.container(border=True):
            st.caption(f"{primary} (the paper's focal method)")
            st.markdown(f"### {cluster_tag(int(top['Cluster']), top['Segment'])}")
            st.markdown(f"{d['Description'].capitalize()}.")
            st.markdown(f"**Suggested action:** {d['Suggested action']}.")
            st.caption(f"{int(d['Customers']):,} customers ({d['% of Customers']:.1f} %) are in this cluster; "
                       f"mean R {d['Mean Recency']:.0f} days, F {d['Mean Frequency']:.1f}, M £{d['Mean Monetary']:,.0f}.")
        st.markdown("**All five algorithms**")
        st.dataframe(results, hide_index=True)
        probabilities = segmenter().gmm_probabilities([[r, f, mo]])[0]
        gmm_names = cluster_descriptions("GMM")["Segment"].tolist()
        st.markdown("**GMM is a soft method: probability of each component**")
        fig = go.Figure(go.Bar(x=[f"C{i} · {n}" for i, n in enumerate(gmm_names)], y=probabilities,
                               marker_color=CLUSTER_COLORS[:len(probabilities)], text=[f"{p:.0%}" for p in probabilities],
                               textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, 1.15], tickformat=".0%"))
        show(styled(fig, 300, legend=False))
        warnings = core.out_of_range({"Recency": r, "Frequency": f, "Monetary": mo},
                                     {k: (v["Lower Bound"], v["Upper Bound"]) for k, v in m["iqr_bounds"].items()})
        if warnings:
            st.warning("This customer would have been removed as an **outlier** before clustering ("
                       + "; ".join(warnings) + "). The assignment is shown, but no such customers shaped the clusters.",
                       icon=":material/warning:")
    with right:
        show(cluster_3d(data, data[m["models"][primary]["column"]], segmenter().centers(primary),
                                   CENTRE_NAMES[primary], highlight=(r, f, mo), height=600), key="segment3d")
        st.caption(f"{primary} clusters of the {len(data):,} customers; the red diamond is the customer you entered.")


def page_clv():
    m = meta()
    s = m["clv"]
    clv = clv_models()
    cutoff = pd.Timestamp(s["cutoff"])
    st.title("Customer lifetime value (CLV) prediction")
    st.markdown(
        f"Predicts each customer's **net spending in the {s['prediction_days']} days from {cutoff:%d %b %Y}** "
        f"(purchases minus cancellations) from their behaviour **before** that date, plus their K-Means++ segment. "
        f"Models: **Random Forest** and **Gradient Boosting**, tuned with RandomizedSearchCV "
        f"({s['search']['iterations']} candidates × {s['search']['cv_folds']}-fold CV).")
    windows = st.columns(3)
    windows[0].metric("Observation window (inputs)", f"{s['observation_days']} days",
                      f"{pd.Timestamp(s['data_start']):%d %b %Y} – {cutoff - pd.Timedelta(days=1):%d %b %Y}",
                      delta_color="off", delta_arrow="off", border=True)
    windows[1].metric("Prediction window (target)", f"{s['prediction_days']} days",
                      f"{cutoff:%d %b %Y} – {pd.Timestamp(s['data_end']):%d %b %Y}",
                      delta_color="off", delta_arrow="off", border=True)
    windows[2].metric("Customers (train / test)", f"{len(clv.train):,} / {len(clv.test):,}", "80 / 20 split",
                      delta_color="off", delta_arrow="off", border=True)

    tab_predict, tab_performance, tab_paper = st.tabs(["Predict for a customer", "Model performance",
                                                       "Comparison with the paper"])
    with tab_predict:
        medians = clv.train[core.RFM].median()
        init_inputs("clv", (int(medians["Recency"]), int(medians["Frequency"]), round(float(medians["Monetary"]), 2)))
        example_buttons("clv", {"Frequent, high spender": (8, 9, 3000.0), "Typical customer": (45, 2, 600.0),
                                "One-off buyer": (200, 1, 250.0)})
        test = clv.test.set_index("CustomerID")
        st.selectbox("…or load a test customer (never used for training, so the true value is known)",
                     test.index.tolist(), index=None, key="clv_pick", placeholder="Type or choose a test customer ID",
                     on_change=load_customer, args=("clv", "clv_test"))
        r, f, mo = rfm_inputs("clv", f"Recency (days before {cutoff:%d %b %Y})", 1000)

        prediction = clv.predict([[r, f, mo]]).iloc[0]
        segment = int(prediction["Segment"])
        seg = clv_segment_descriptions().set_index("Cluster").loc[segment]
        rf, gb = prediction["Random Forest (Tuned)"], prediction["Gradient Boosting (Tuned)"]
        history = clv.same_length_history(mo)
        cards = st.columns(4)
        cards[0].metric("Segment (K-Means++)", f"Cluster {segment}", seg["Segment"], delta_color="off",
                        delta_arrow="off", border=True)
        cards[1].metric("Random Forest (tuned)", money(max(rf, 0.0)), border=True,
                        help="Predicted net spend in the next 100 days")
        cards[2].metric("Gradient Boosting (tuned)", money(max(gb, 0.0)), border=True,
                        help="Predicted net spend in the next 100 days")
        cards[3].metric("Same-length history", money(history), border=True,
                        help=f"Spend before the cutoff rescaled to {s['prediction_days']} days "
                             f"(× {s['prediction_days']}/{s['observation_days']}): the notebook's 'historical CLV'")

        loaded = st.session_state.get("clv_loaded")
        if loaded is not None and loaded in test.index:
            row = test.loc[loaded]
            if (int(row["Recency"]), int(row["Frequency"]), float(row["Monetary"])) == (r, f, mo):
                st.success(f"Test customer {loaded}: actual net spend in the prediction window = "
                           f"**{money(row['CLV_Target'])}** (the models never saw this customer).",
                           icon=":material/check_circle:")
        label = "High" if gb >= history else "Low"
        st.markdown(f"Paper-style label (Gradient Boosting vs same-length history): **{label}** CLV — the model "
                    f"expects this customer to spend {'more' if label == 'High' else 'less'} than in an equally long "
                    f"period before the cutoff.")

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("**Predicted net spend in the next 100 days, all four models**")
            names = [f"{n} ({v})" for n, v in core.CLV_MODELS]
            values = [max(float(prediction[n]), 0.0) for n in names]
            fig = go.Figure(go.Bar(x=names, y=values, marker_color=["#A9B7D6", "#4A6FA5", "#F2B8B8", RED],
                                   text=[money(v) for v in values], textposition="outside"))
            fig.add_hline(y=history, line_dash="dash", line_color=GRAY,
                          annotation_text="same-length history", annotation_position="top left")
            fig.update_layout(yaxis=dict(tickprefix="£", range=[0, max(values + [history]) * 1.25 + 1]))
            show(styled(fig, 360, legend=False))
        with right:
            st.markdown(f"**Segment {segment} · {seg['Segment']}**: {seg['Description']}.")
            st.caption("The segment is an input of both models (one-hot), from a K-Means++ model fitted on the "
                       "training customers only.")
            warnings = core.out_of_range({"Recency": r, "Frequency": f, "Monetary": mo}, clv.training_bounds())
            if warnings:
                st.warning("Outside the range of the training customers, so the prediction is an extrapolation: "
                           + "; ".join(warnings) + ".", icon=":material/warning:")
            st.caption("Negative model outputs are shown as £0, because the target (net spend) is never negative.")

    with tab_performance:
        metrics = clv.evaluate()
        st.dataframe(metrics, hide_index=True,
                     column_config={"MAE": st.column_config.NumberColumn("MAE (£)", format="%.2f"),
                                    "MSE": st.column_config.NumberColumn(format="%.2f"),
                                    "RMSE": st.column_config.NumberColumn("RMSE (£)", format="%.2f"),
                                    "R²": st.column_config.NumberColumn(format="%.4f")})
        st.caption(f"Computed now on the {len(clv.test):,} test customers, which were never used for training or "
                   "tuning. Identical to the notebook's Section 24 table.")
        left, right = st.columns([3, 2], gap="large")
        with left:
            choice = st.segmented_control("Model", [f"{n} ({v})" for n, v in core.CLV_MODELS],
                                          default="Gradient Boosting (Tuned)", required=True, key="avp_model",
                                          format_func=lambda v: f"{'RF' if v.startswith('Random') else 'GB'} · "
                                                                f"{'tuned' if v.endswith('(Tuned)') else 'untuned'}")
            zoom = st.toggle("Zoom: actual value ≤ 99th percentile", value=True)
            st.markdown(f"**{choice}: actual vs predicted** (test customers)")
            predictions = clv.test_predictions()
            y, p = predictions["CLV_Target"], predictions[choice]
            limit = float(np.quantile(y, 0.99)) if zoom else float(max(y.max(), p.max()))
            mask = y <= limit
            fig = go.Figure()
            fig.add_trace(go.Scattergl(x=y[mask], y=p[mask], mode="markers", name="Test customers",
                                       marker=dict(size=6, color="#1f77b4", opacity=0.5),
                                       customdata=predictions.loc[mask, ["CustomerID"]],
                                       hovertemplate="Customer %{customdata[0]}<br>actual £%{x:,.2f}<br>"
                                                     "predicted £%{y:,.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, limit], y=[0, limit], mode="lines", name="Perfect prediction",
                                     line=dict(color=RED, dash="dash")))
            fig.update_layout(xaxis_title="Actual net spend (£)", yaxis_title="Predicted net spend (£)")
            show(styled(fig, 420))
        with right:
            importance = clv.feature_importance()
            st.markdown("**Which inputs do the tuned models use?**")
            fig = go.Figure()
            for name, color in [("Random Forest", "#4A6FA5"), ("Gradient Boosting", RED)]:
                part = importance[importance["Model"] == name]
                fig.add_trace(go.Bar(y=part["Feature"], x=part["Importance"], orientation="h", name=name,
                                     marker_color=color))
            fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"),
                              xaxis_title="Impurity-based importance")
            show(styled(fig, 360))
            params = []
            for name, values in s["tuned_params"].items():
                for key, value in values.items():
                    params.append({"Model": name, "Hyperparameter": key,
                                   "Best value": f"{value:.4f}" if isinstance(value, float) else str(value)})
            st.markdown("**Best hyperparameters (RandomizedSearchCV)**")
            st.dataframe(pd.DataFrame(params), hide_index=True)
            st.caption("Best cross-validated R² on the training customers: " + ", ".join(
                f"{k} {v:.4f}" for k, v in s["best_cv_r2"].items()))
        st.markdown("**Paper-style CLV per segment** (test customers, tuned models)")
        per_cluster = table("paper_style_clv_per_cluster.csv")
        st.dataframe(per_cluster, hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in per_cluster.columns if c not in ("Cluster", "Test customers",
                                                                              "Paper-style CLV label (GB)")})

    with tab_paper:
        paper = table("paper_vs_ours_clv.csv")
        shown = pd.DataFrame({"Model": paper["Model"], "Tuned/Untuned": paper["Tuned/Untuned"],
                              "Reference Paper Result: MAE": paper["Paper MAE"],
                              "Our Reproduction Result: MAE (£)": paper["Our MAE"],
                              "Reference Paper Result: R²": paper["Paper R²"],
                              "Our Reproduction Result: R²": paper["Our R²"]})
        st.dataframe(shown, hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.4f" if "R²" in c else "%.2f")
                                    for c in shown.columns[2:]})
        with st.expander("Full table, including MSE and RMSE"):
            st.dataframe(paper.rename(columns=lambda c: c.replace("Paper ", "Reference Paper Result: ")
                                      .replace("Our ", "Our Reproduction Result: ")), hide_index=True)
        st.markdown(
            "- The paper reports **R² ≈ 0.9999** for every model but does not say how its CLV target was built.\n"
            f"- Our target is **future** net spending after {cutoff:%d %b %Y}, predicted only from behaviour before "
            "that date, so no future information leaks into the inputs. Future spending is genuinely hard to "
            "predict from RFM alone, which is why our R² is about 0.15–0.18.\n"
            "- The notebook's Section 26.4 shows the effect of an **in-period** target (computed from the same "
            "period as the inputs): R² becomes 0.9999–1.0000, because the model only re-computes a known "
            "quantity. That illustration is not one of our results.")


def page_lookup():
    m = meta()
    primary = m["primary_model"]
    clv = clv_models()
    data, raw = customers(), table("rfm_raw.csv")
    st.title("Customer lookup and batch prediction")
    tab_one, tab_batch = st.tabs(["Look up a customer", "Batch prediction (CSV)"])

    with tab_one:
        ids = sorted(set(raw["CustomerID"]) | set(clv.train["CustomerID"]) | set(clv.test["CustomerID"]))
        segmented = set(data["CustomerID"])
        default = next(i for i in clv.test["CustomerID"] if i in segmented)
        cid = st.selectbox("Customer ID", ids, index=ids.index(default), key="lookup_id")
        left, right = st.container(), st.container()
        with left:
            st.subheader("Segmentation (whole period)")
            if cid in set(data["CustomerID"]):
                row = data.set_index("CustomerID").loc[cid]
                cards = st.columns(3)
                cards[0].metric("Recency", f"{int(row['Recency'])} days", border=True)
                cards[1].metric("Frequency", f"{int(row['Frequency'])} orders", border=True)
                cards[2].metric("Monetary", money(row["Monetary"]), border=True)
                rows = []
                for name in core.MODEL_ORDER:
                    cluster = int(row[m["models"][name]["column"]])
                    d = cluster_descriptions(name).set_index("Cluster").loc[cluster]
                    rows.append({"Algorithm": name, "Cluster": cluster, "Segment": d["Segment"],
                                 "Suggested action": d["Suggested action"]})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            else:
                row = raw.set_index("CustomerID").loc[cid] if cid in set(raw["CustomerID"]) else None
                if row is not None:
                    reasons = core.out_of_range({f: row[f] for f in core.RFM},
                                                {k: (v["Lower Bound"], v["Upper Bound"])
                                                 for k, v in m["iqr_bounds"].items()})
                    st.warning(f"Removed as an outlier by the IQR rule ({'; '.join(reasons)}), so it is not part of "
                               "the segmentation.", icon=":material/warning:")
                else:
                    st.info("No purchases in the cleaned data.")
        with right:
            st.subheader(f"CLV (before / after {pd.Timestamp(m['clv']['cutoff']):%d %b %Y})")
            in_test, in_train = cid in set(clv.test["CustomerID"]), cid in set(clv.train["CustomerID"])
            if in_test or in_train:
                source = clv.test if in_test else clv.train
                row = source.set_index("CustomerID").loc[cid]
                prediction = clv.predict([[row["Recency"], row["Frequency"], row["Monetary"]]]).iloc[0]
                cards = st.columns(3)
                cards[0].metric("Actual net spend (next 100 days)", money(row["CLV_Target"]), border=True)
                cards[1].metric("Predicted: Random Forest (tuned)",
                                money(max(prediction["Random Forest (Tuned)"], 0.0)), border=True)
                cards[2].metric("Predicted: Gradient Boosting (tuned)",
                                money(max(prediction["Gradient Boosting (Tuned)"], 0.0)), border=True)
                st.caption(f"Before the cutoff: R {int(row['Recency'])} days, F {int(row['Frequency'])}, "
                           f"M {money(row['Monetary'])} · segment {int(row['Cluster'])}.")
                if in_test:
                    st.success("Test customer: never used for training or tuning (a fair check).",
                               icon=":material/check_circle:")
                else:
                    st.info("Training customer: the models learned from this customer, so the prediction is "
                            "in-sample.", icon=":material/info:")
            else:
                st.info("Not in the CLV data: no purchases before the cutoff, or removed by the CLV outlier rule.")

    with tab_batch:
        st.markdown("Upload a CSV with the columns **Recency, Frequency, Monetary** (an optional **CustomerID** is "
                    "kept). Every row gets a cluster from all five algorithms and a CLV prediction from the tuned "
                    "models.")
        template = pd.DataFrame({"CustomerID": [1001, 1002, 1003], "Recency": [12, 30, 250], "Frequency": [8, 1, 1],
                                 "Monetary": [2500.0, 250.0, 300.0]})
        st.download_button("Download a template CSV", template.to_csv(index=False).encode("utf-8"),
                           "rfm_template.csv", "text/csv", icon=":material/download:")
        uploaded = st.file_uploader("CSV file (up to 5 MB)", type=["csv"], key="batch_file")
        if uploaded is None:
            use_example = st.toggle(f"Try it with the {len(clv.test):,} test customers", value=True)
            if not use_example:
                return
            frame, source = clv.test[["CustomerID"] + core.RFM], "the test customers"
        else:
            try:
                frame, source = pd.read_csv(uploaded, nrows=50_000), uploaded.name
            except Exception as error:                     # malformed file: tell the user, do not crash the app
                st.error(f"Could not read the file as CSV ({type(error).__name__}).")
                return
        try:
            result, skipped = batch_predict(frame)
        except ValueError as error:
            st.error(str(error))
            return
        st.success(f"{len(result):,} customers from {source} scored"
                   + (f"; {skipped:,} rows skipped (missing or invalid values)." if skipped else "."),
                   icon=":material/check_circle:")
        st.dataframe(result, hide_index=True, height=360)
        st.download_button("Download the predictions", result.to_csv(index=False).encode("utf-8"),
                           "segment_and_clv_predictions.csv", "text/csv", type="primary", icon=":material/download:")
        summary = (result.groupby(f"{primary} segment")
                   .agg(Customers=("Recency", "size"),
                        **{"Mean predicted CLV (GB)": ("Predicted CLV, Gradient Boosting (£)", "mean")})
                   .reset_index())
        st.dataframe(summary, hide_index=True,
                     column_config={"Mean predicted CLV (GB)": st.column_config.NumberColumn(format="%.2f")})


def batch_predict(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Cluster + CLV predictions for an uploaded table. Raises ValueError with a readable message."""
    out, skipped = core.clean_rfm_table(frame)
    m = meta()
    primary = m["primary_model"]
    clusters = segmenter().predict_all(out)
    names = cluster_descriptions(primary).set_index("Cluster")["Segment"]
    out[f"{primary} segment"] = clusters[primary].map(names)
    for name in core.MODEL_ORDER:
        out[f"{name} cluster"] = clusters[name]
    predicted = clv_models().predict(out)
    out["Predicted CLV, Random Forest (£)"] = predicted["Random Forest (Tuned)"].clip(lower=0).round(2)
    out["Predicted CLV, Gradient Boosting (£)"] = predicted["Gradient Boosting (Tuned)"].clip(lower=0).round(2)
    return out, skipped


# ================================================================================================ layout
st.logo(str(APP_DIR / "assets" / "sastra_logo.png"), size="large")
navigation = st.navigation({
    "Project": [st.Page(page_overview, title="Overview", icon=":material/home:", default=True),
                st.Page(page_data, title="Data & RFM", icon=":material/table_chart:", url_path="data")],
    "Clustering": [st.Page(page_clustering, title="Clustering results", icon=":material/scatter_plot:",
                           url_path="clustering"),
                   st.Page(page_compare, title="Compare algorithms", icon=":material/compare_arrows:",
                           url_path="compare"),
                   st.Page(page_segment, title="Segment a customer", icon=":material/person_search:",
                           url_path="segment")],
    "Prediction": [st.Page(page_clv, title="CLV prediction", icon=":material/trending_up:", url_path="clv"),
                   st.Page(page_lookup, title="Lookup & batch", icon=":material/manage_search:", url_path="lookup")],
}, expanded=True)
with st.sidebar:
    st.markdown("**Customer Segmentation & CLV Prediction**")
    st.caption("127018042 · Pattan Sameera Hussainy  \nGuide: Dr. Vidivelli S  \nSASTRA Deemed University")
    st.caption("Data: UCI Online Retail (Dec 2010 – Dec 2011)  \nBase paper: Ling & Weiling, IEEE Access 2025")
navigation.run()
