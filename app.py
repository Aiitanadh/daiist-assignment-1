"""
Assignment 1 - Gradio dashboard (run with: uv run python main.py app)

Nothing is trained here. At startup the app LOADS:
  - the three saved models from models/
  - the fitted FeatureBuilder (feature engineering learned on train only)
  - the split assignment + metrics written by train.py
and then just runs predictions on the validation/test products.
"""

import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from features import category_group, listing_order
from modeling import (
    ARTIFACTS_DIR, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, MODEL_NAMES, ROOT,
    business_cost, confusion_at, load_models, naive_policy_cost, predict_proba, summary_metrics,
)

# ---------------------------------------------------------------------------
# Load everything once (no training)
# ---------------------------------------------------------------------------
for required in ["feature_builder.joblib", "metrics.json", "splits.csv", "coefficients.csv"]:
    if not (ARTIFACTS_DIR / required).exists():
        raise SystemExit(f"Missing artifacts/{required}. Run `uv run python main.py train` first.")

MODELS = load_models()
BUILDER = joblib.load(ARTIFACTS_DIR / "feature_builder.joblib")
METRICS = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
COEFS = pd.read_csv(ARTIFACTS_DIR / "coefficients.csv", index_col="feature")
FEATS = pd.read_csv(ARTIFACTS_DIR / "engineered_features.csv")

RAW = pd.read_csv(ROOT / "data" / "product_info.csv")
SPLITS = pd.read_csv(ARTIFACTS_DIR / "splits.csv")
DATA = RAW.merge(SPLITS, on="product_id", how="inner").set_index("product_id", drop=False)
FEATS = FEATS.merge(SPLITS[["product_id", "split", "hit"]], on="product_id")

PARTS = {s: DATA[DATA["split"] == s] for s in ["train", "val", "test"]}
Y = {s: PARTS[s]["hit"].to_numpy() for s in ["val", "test"]}
X = {s: BUILDER.transform(PARTS[s]) for s in ["val", "test"]}
PROBS = {s: {m: predict_proba(m, MODELS[m], X[s]) for m in MODEL_NAMES} for s in ["val", "test"]}

COLORS = {"scikit-learn": "#B0305A", "Manual PyTorch": "#3F6C9E", "nn.Module PyTorch": "#C9952B"}
INK, MUTED, BLUSH = "#2E1A33", "#7A6A80", "#F6E6EA"
LAYOUT = dict(template="plotly_white", font=dict(family="Instrument Sans, Helvetica, Arial", color=INK),
              margin=dict(l=60, r=20, t=50, b=50), legend=dict(orientation="h", y=-0.2))

THRESHOLDS = METRICS["thresholds_from_val"]


# ---------------------------------------------------------------------------
# Tab 1: compare the three models
# ---------------------------------------------------------------------------
def metrics_table() -> pd.DataFrame:
    rows = []
    for m in MODEL_NAMES:
        r = summary_metrics(Y["test"], PROBS["test"][m], THRESHOLDS[m])
        rows.append({"model": m, **r})
    prior = METRICS["train_hit_rate"]
    base = summary_metrics(Y["test"], np.full(len(Y["test"]), prior), 0.5)
    base["business_cost"] = min(naive_policy_cost(Y["test"]).values())
    rows.append({"model": "Naive baseline", **base})
    df = pd.DataFrame(rows)[["model", "roc_auc", "pr_auc", "log_loss", "brier", "threshold",
                             "precision", "recall", "business_cost"]]
    df = df.rename(columns={"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "log_loss": "log-loss",
                            "brier": "Brier", "threshold": "threshold (from val)",
                            "business_cost": "test cost ($)"})
    return df.round(4)


def agreement_table() -> pd.DataFrame:
    df = pd.DataFrame(METRICS["agreement"]).T
    df.columns = ["max |Δ prob|", "max |Δ coef|", "Δ intercept", "coef. correlation"]
    df.insert(0, "compared with scikit-learn", df.index)
    return df.map(lambda v: f"{v:.2e}" if isinstance(v, float) else v)


def prediction_vs_actual_plot() -> go.Figure:
    """Classification version of 'prediction vs actual': a calibration plot
    (predicted probability vs the share of products that really were hits)
    plus the spread of predictions for actual hits vs non-hits."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Predicted P(hit) vs actual hit rate (test, 10 bins)",
        "Predicted P(hit) for actual non-hits vs hits"))
    y = Y["test"]
    for m in MODEL_NAMES:
        p = PROBS["test"][m]
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        fig.add_trace(go.Scatter(x=cal["p"], y=cal["y"], mode="lines+markers", name=m,
                                 line=dict(color=COLORS[m], width=2), legendgroup=m), 1, 1)
        fig.add_trace(go.Box(x=np.where(y == 1, "actual hit", "actual non-hit"), y=p, name=m,
                             marker_color=COLORS[m], legendgroup=m, showlegend=False,
                             boxpoints=False), 1, 2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                             line=dict(color=MUTED, dash="dot")), 1, 1)
    fig.update_layout(**LAYOUT, height=430, boxmode="group")
    fig.update_xaxes(title_text="mean predicted P(hit)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="actual hit rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="predicted P(hit)", row=1, col=2)
    return fig


def agreement_scatter() -> go.Figure:
    fig = go.Figure()
    ref = PROBS["test"]["scikit-learn"]
    for m in MODEL_NAMES[1:]:
        fig.add_trace(go.Scattergl(x=ref, y=PROBS["test"][m], mode="markers", name=m,
                                   marker=dict(color=COLORS[m], size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="identical",
                             line=dict(color=MUTED, dash="dot")))
    fig.update_layout(**LAYOUT, height=400, title="Each test product: PyTorch P(hit) vs scikit-learn P(hit)",
                      xaxis_title="scikit-learn", yaxis_title="PyTorch version")
    return fig


def coefficient_plot(top_n: int) -> go.Figure:
    c = COEFS.drop(index="(intercept)")
    order = c["scikit-learn"].abs().sort_values(ascending=False).head(int(top_n)).index[::-1]
    fig = go.Figure()
    for m in MODEL_NAMES:
        fig.add_trace(go.Bar(y=list(order), x=c.loc[order, m], name=m, orientation="h",
                             marker_color=COLORS[m]))
    fig.update_layout(**LAYOUT, barmode="group", height=160 + 34 * int(top_n),
                      title="Weights on standardised features (bigger = stronger effect on the log-odds)",
                      xaxis_title="coefficient")
    fig.update_yaxes(dtick=1, automargin=True)  # show every feature name
    return fig


def loss_curve_plot() -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Manual loop: every learning rate tried", "nn.Module + Adam: every learning rate tried"))
    for col, m in enumerate(["Manual PyTorch", "nn.Module PyTorch"], start=1):
        runs = METRICS["loss_curves_all_lr"][m]
        chosen = str(METRICS["lr_manual" if col == 1 else "lr_nn"])
        for lr, hist in runs.items():
            h = pd.DataFrame(hist)
            fig.add_trace(go.Scatter(
                x=h["epoch"], y=h["train"], mode="lines",
                name=f"{m.split()[0]} lr={lr}" + (" (chosen)" if lr == chosen else ""),
                line=dict(width=3 if lr == chosen else 1.2,
                          color=COLORS[m] if lr == chosen else None)), 1, col)
    fig.update_layout(**LAYOUT, height=420)
    fig.update_yaxes(title_text="training objective (BCE + L2)", type="log")
    fig.update_xaxes(title_text="epoch")
    return fig


# ---------------------------------------------------------------------------
# Tab 2: threshold + business cost
# ---------------------------------------------------------------------------
def cost_view(model: str, threshold: float, cost_fp: float, cost_fn: float, split: str):
    y, p = Y[split], PROBS[split][model]
    cm = confusion_at(y, p, threshold)
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    labels = [[f"True negative<br>{cm['tn']}<br>not promoted, not a hit",
               f"False positive<br>{cm['fp']}<br>promoted, flopped (${cost_fp:,.0f} each)"],
              [f"False negative<br>{cm['fn']}<br>missed hit (${cost_fn:,.0f} each)",
               f"True positive<br>{cm['tp']}<br>promoted, became a hit"]]
    cm_fig = go.Figure(go.Heatmap(
        z=z, x=["predicted: not a hit", "predicted: hit (promote)"],
        y=["actual: not a hit", "actual: hit"], text=labels, texttemplate="%{text}",
        colorscale=[[0, "#FFFFFF"], [1, "#E7A9B9"]], showscale=False))
    cm_fig.update_layout(**LAYOUT, height=380, title=f"Confusion matrix at threshold {threshold:.2f}")
    cm_fig.update_yaxes(autorange="reversed")

    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [business_cost(y, p, t, cost_fp, cost_fn) for t in grid]
    naive = naive_policy_cost(y, cost_fp, cost_fn)
    cost = business_cost(y, p, threshold, cost_fp, cost_fn)
    curve = go.Figure()
    curve.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name=f"{model}",
                               line=dict(color=COLORS[model], width=2.5)))
    for label, value, dash, pos in [("promote nothing", naive["promote nothing"], "dash", "top left"),
                                    ("promote everything", naive["promote everything"], "dot", "bottom right")]:
        curve.add_hline(y=value, line=dict(color=MUTED, dash=dash),
                        annotation_text=f"{label}: ${value:,.0f}", annotation_position=pos)
    curve.add_vline(x=threshold, line=dict(color=INK, width=1))
    curve.add_trace(go.Scatter(x=[threshold], y=[cost], mode="markers", name="your threshold",
                               marker=dict(color=INK, size=11)))
    curve.update_layout(**LAYOUT, height=380, title="Total cost at every threshold",
                        xaxis_title="threshold (promote if P(hit) ≥ threshold)", yaxis_title="cost ($)")

    best_naive = min(naive, key=naive.get)
    best_t = float(grid[int(np.argmin(costs))])
    theory_t = cost_fp / (cost_fp + cost_fn)
    promoted = cm["tp"] + cm["fp"]
    summary = (
        f"### Cost on the {split} set: **${cost:,.0f}**\n"
        f"{promoted} of {len(y)} products promoted. Best no-model policy (*{best_naive}*): "
        f"${naive[best_naive]:,.0f}, so the model saves **${naive[best_naive] - cost:,.0f}** "
        f"({(naive[best_naive] - cost) / naive[best_naive]:.0%}).\n\n"
        f"Cheapest threshold on this set: {best_t:.2f}. Theory for a perfectly calibrated model: "
        f"cost_FP / (cost_FP + cost_FN) = {theory_t:.2f}. "
        f"Threshold chosen on validation for {model}: {THRESHOLDS[model]:.2f}."
    )
    return cm_fig, curve, summary


# ---------------------------------------------------------------------------
# Tab 3: data
# ---------------------------------------------------------------------------
FEATURE_CHOICES = [c for c in FEATS.columns if c not in ("product_id", "split", "hit")]


def feature_plot(feature: str, split: str) -> go.Figure:
    d = FEATS if split == "all" else FEATS[FEATS["split"] == split]
    fig = go.Figure()
    for value, name, color in [(0, "not a hit", "#9C8AA5"), (1, "hit", "#B0305A")]:
        fig.add_trace(go.Histogram(x=d.loc[d["hit"] == value, feature], name=name,
                                   marker_color=color, opacity=0.65, histnorm="probability",
                                   nbinsx=40))
    fig.update_layout(**LAYOUT, barmode="overlay", height=400,
                      title=f"{feature}: distribution for hits vs non-hits ({split})",
                      xaxis_title=feature, yaxis_title="share of products")
    return fig


def target_plots():
    d = DATA.copy()
    d["log10 loves"] = np.log10(d["loves_count"].clip(lower=1))
    loves = go.Figure()
    for value, name, color in [(0, "not a hit", "#9C8AA5"), (1, "hit", "#B0305A")]:
        loves.add_trace(go.Histogram(x=d.loc[d["hit"] == value, "log10 loves"], name=name,
                                     marker_color=color, opacity=0.65, nbinsx=50))
    loves.update_layout(**LAYOUT, barmode="overlay", height=380,
                        title="Target: loves (log10) — hits overlap non-hits because 'hit' is relative",
                        xaxis_title="log10(loves_count)", yaxis_title="products")

    full = RAW.copy()
    full["listing decile"] = pd.qcut(listing_order(full), 10, labels=[f"D{i}" for i in range(1, 11)])
    med = full.groupby("listing decile", observed=True)["loves_count"].median()
    age = go.Figure(go.Bar(x=med.index.astype(str), y=med.values, marker_color="#3F6C9E"))
    age.update_layout(**LAYOUT, height=380,
                      title="Why the target is time-relative: median loves by listing decile (D1 = oldest)",
                      xaxis_title="listing-order decile (product ID rank)", yaxis_title="median loves")

    rates = d.groupby(category_group(d))["hit"].mean().sort_values()
    cats = go.Figure(go.Bar(x=rates.values, y=rates.index, orientation="h", marker_color="#C9952B"))
    cats.update_layout(**LAYOUT, height=360, title="Hit rate per category (≈25% by construction)",
                       xaxis_title="hit rate", xaxis_range=[0, 0.5])
    return loves, age, cats


# ---------------------------------------------------------------------------
# Tab 4: score a product that isn't in the data
# ---------------------------------------------------------------------------
TRAIN_RAW = PARTS["train"]
BRANDS = sorted(TRAIN_RAW["brand_name"].unique().tolist())
CATEGORIES = sorted(TRAIN_RAW["primary_category"].unique().tolist())
SUBCATS = sorted(TRAIN_RAW["secondary_category"].dropna().unique().tolist())


def score_product(name, brand, new_brand, category, subcategory, price, size, n_variants,
                  variation, limited, online_only, exclusive):
    row = pd.DataFrame([{
        "product_name": name or "", "brand_name": "__new_brand__" if new_brand else brand,
        "price_usd": float(price), "size": size or None, "child_count": int(n_variants),
        "child_max_price": np.nan, "child_min_price": np.nan, "value_price_usd": np.nan,
        "limited_edition": int(limited), "online_only": int(online_only),
        "sephora_exclusive": int(exclusive), "ingredients": None, "highlights": None,
        "primary_category": category, "secondary_category": subcategory,
        "variation_type": None if variation == "None" else variation,
    }])
    Xr = BUILDER.transform(row)
    lines = ["| model | P(hit) | decision at its validation threshold |", "|---|---|---|"]
    for m in MODEL_NAMES:
        p = float(predict_proba(m, MODELS[m], Xr)[0])
        decision = "promote" if p >= THRESHOLDS[m] else "don't promote"
        lines.append(f"| {m} | {p:.3f} | {decision} (≥ {THRESHOLDS[m]:.2f}) |")
    raw = BUILDER.transform_raw(row).iloc[0]
    lines.append(f"\nBrand hit rate used: {raw['brand_hit_rate']:.2f} · price vs typical "
                 f"{subcategory}: {np.exp(raw['price_vs_subcategory']):.2f}×")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&display=swap');
.gradio-container {max-width: 1200px !important; margin: auto;}
#title h1 {font-family: 'Fraunces', Georgia, serif; font-weight: 600; color: #2E1A33; margin-bottom: 0;}
#title p {color: #7A6A80; max-width: 75ch;}
"""
THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(c50="#FBF0F3", c100="#F6E6EA", c200="#EFC7D2", c300="#E7A9B9",
                                c400="#D46F8C", c500="#B0305A", c600="#982A4E", c700="#7C2240",
                                c800="#5F1A31", c900="#431222", c950="#2E0C17"),
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Instrument Sans"), "Helvetica", "Arial", "sans-serif"],
)

n = METRICS["n_rows"]
with gr.Blocks(title="Sephora hit predictor") as demo:
    gr.Markdown(
        "# Will this new listing become a Sephora hit?\n"
        "Logistic regression trained three ways (scikit-learn, a manual PyTorch loop, "
        "nn.Module + torch.optim) to flag new products that will end up in the top 25% of "
        "loves for their category and listing period, using only what is known on listing day. "
        f"Time-ordered split: {n['train']:,} oldest products to train, {n['val']:,} to tune, "
        f"{n['test']:,} newest to test. Models are loaded from disk; nothing is retrained here. "
        "(The three models' lines sit on top of each other in most plots: that's the point.)",
        elem_id="title")

    with gr.Tab("Compare the models"):
        gr.Markdown("### Test-set results (the naive baseline always predicts the training hit rate)")
        gr.Dataframe(metrics_table(), interactive=False)
        gr.Markdown("### Do the three implementations agree?")
        gr.Dataframe(agreement_table(), interactive=False)
        gr.Plot(prediction_vs_actual_plot())
        gr.Plot(agreement_scatter())
        top_n = gr.Slider(5, len(COEFS) - 1, value=15, step=1, label="Number of features to show")
        coef_fig = gr.Plot(coefficient_plot(15))
        top_n.change(coefficient_plot, top_n, coef_fig)
        gr.Plot(loss_curve_plot())

    with gr.Tab("Threshold and cost"):
        gr.Markdown(
            "Each promoted product that flops wastes the promotion budget (false positive). "
            "Each hit you didn't promote loses its launch-period margin (false negative). "
            "Both costs are assumptions; change them and watch the best threshold move.")
        with gr.Row():
            model_dd = gr.Dropdown(MODEL_NAMES, value="scikit-learn", label="Model")
            split_dd = gr.Radio(["test", "val"], value="test", label="Products")
        with gr.Row():
            thr = gr.Slider(0.01, 0.99, value=THRESHOLDS["scikit-learn"], step=0.01,
                            label="Decision threshold: promote if P(hit) ≥ this")
            cfp = gr.Number(value=COST_FALSE_POSITIVE, label="Cost of a false positive ($)")
            cfn = gr.Number(value=COST_FALSE_NEGATIVE, label="Cost of a false negative ($)")
        cost_md = gr.Markdown()
        with gr.Row():
            cm_plot = gr.Plot()
            curve_plot = gr.Plot()
        inputs = [model_dd, thr, cfp, cfn, split_dd]
        for comp in inputs:
            comp.change(cost_view, inputs, [cm_plot, curve_plot, cost_md])
        demo.load(cost_view, inputs, [cm_plot, curve_plot, cost_md])

    with gr.Tab("Explore the data"):
        loves_fig, age_fig, cat_fig = target_plots()
        gr.Plot(age_fig)
        gr.Plot(loves_fig)
        gr.Plot(cat_fig)
        gr.Markdown("### Engineered features (unscaled), hits vs non-hits")
        with gr.Row():
            feat_dd = gr.Dropdown(FEATURE_CHOICES, value="brand_hit_rate", label="Feature")
            feat_split = gr.Radio(["train", "val", "test", "all"], value="train", label="Split")
        feat_fig = gr.Plot(feature_plot("brand_hit_rate", "train"))
        feat_dd.change(feature_plot, [feat_dd, feat_split], feat_fig)
        feat_split.change(feature_plot, [feat_dd, feat_split], feat_fig)

    with gr.Tab("Score a new product"):
        gr.Markdown("Describe a product as a buyer would see it on listing day.")
        with gr.Row():
            p_name = gr.Textbox(value="Glow Serum Foundation", label="Product name")
            p_brand = gr.Dropdown(BRANDS, value="Charlotte Tilbury" if "Charlotte Tilbury" in BRANDS else BRANDS[0],
                                  label="Brand")
            p_new = gr.Checkbox(value=False, label="Brand is new to Sephora")
        with gr.Row():
            p_cat = gr.Dropdown(CATEGORIES, value="Makeup", label="Category")
            p_sub = gr.Dropdown(SUBCATS, value="Face" if "Face" in SUBCATS else SUBCATS[0],
                                label="Sub-category")
            p_price = gr.Number(value=38, label="Price (USD)")
            p_size = gr.Textbox(value="1 oz/ 30 mL", label="Size")
        with gr.Row():
            p_var = gr.Dropdown(["None", "Color", "Size", "Size + Concentration + Formulation"],
                                value="Color", label="Variation type")
            p_n = gr.Slider(0, 60, value=30, step=1, label="Number of shades / variants")
            p_lim = gr.Checkbox(label="Limited edition")
            p_onl = gr.Checkbox(label="Online only")
            p_exc = gr.Checkbox(label="Sephora exclusive")
        score_btn = gr.Button("Score this product", variant="primary")
        score_out = gr.Markdown()
        score_inputs = [p_name, p_brand, p_new, p_cat, p_sub, p_price, p_size, p_n, p_var,
                        p_lim, p_onl, p_exc]
        score_btn.click(score_product, score_inputs, score_out)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, inbrowser=not os.environ.get("CI"))
