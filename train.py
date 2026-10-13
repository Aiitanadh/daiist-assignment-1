"""
Assignment 1 - training pipeline (run with: uv run python main.py train)

Question: will a product newly listed at Sephora become a "hit" (top 25% of
loves among same-category products listed in the same period)?
Decision it supports: which new listings get premium promotion.

Steps
  1. Load data, drop rows whose outcome isn't observable yet, build target.
  2. Time-ordered split (oldest 70% train / next 15% val / newest 15% test).
  3. Fit feature engineering on train only, transform all three splits.
  4. Naive baselines.
  5. Logistic regression three ways (sklearn / manual PyTorch / nn.Module).
     The validation set is used ONLY to pick C, the learning rates and the
     decision threshold. The test set is touched once, at the end.
  6. Compare, then save models + everything the dashboard needs.
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features import FeatureBuilder, category_group, listing_order, make_target
from modeling import (
    ARTIFACTS_DIR, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, MODEL_NAMES, MODELS_DIR, ROOT,
    LogisticRegressionNet, bce_with_l2, best_threshold, business_cost, coefficients, l2_strength,
    naive_policy_cost, predict_proba, summary_metrics,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)  # keeps the PyTorch runs reproducible
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Data + target
# ---------------------------------------------------------------------------
raw = pd.read_csv(ROOT / "data" / "product_info.csv")
log(f"Loaded {len(raw):,} products x {raw.shape[1]} columns")

# 'new' products have only been listed for a few weeks: their loves haven't
# had time to accumulate, so their label isn't known yet (it's censored).
# 'Gifts' are 4 gift cards with 0 loves, not real products.
keep = (raw["new"] == 0) & (raw["primary_category"] != "Gifts")
df = raw[keep].copy()
log(f"Dropped {int((raw['new'] == 1).sum())} 'new' products (label not observable yet) "
    f"and {int((raw['primary_category'] == 'Gifts').sum())} gift cards -> {len(df):,} rows")

df["hit"] = make_target(df)
log(f"Hit rate: {df['hit'].mean():.1%}")

# ---------------------------------------------------------------------------
# 2. Time-ordered split
# ---------------------------------------------------------------------------
df["listing_rank"] = listing_order(df)
df = df.sort_values("listing_rank")
n = len(df)
cut_train, cut_val = int(0.70 * n), int(0.85 * n)
df["split"] = "train"
df.iloc[cut_train:cut_val, df.columns.get_loc("split")] = "val"
df.iloc[cut_val:, df.columns.get_loc("split")] = "test"
train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])
log(f"Split by listing order: train={len(train):,} val={len(val):,} test={len(test):,} "
    f"(hit rates {train['hit'].mean():.1%} / {val['hit'].mean():.1%} / {test['hit'].mean():.1%})")

# ---------------------------------------------------------------------------
# 3. Features (fit on train only)
# ---------------------------------------------------------------------------
builder = FeatureBuilder().fit(train, train["hit"])
X_train = builder.transform(train, training_rows=True)
X_val = builder.transform(val)
X_test = builder.transform(test)
y_train, y_val, y_test = (s["hit"].to_numpy() for s in (train, val, test))
feature_names = builder.feature_names_
log(f"{len(feature_names)} features after engineering")

# ---------------------------------------------------------------------------
# 4. Naive baselines
# ---------------------------------------------------------------------------
prior = float(y_train.mean())
p_baseline_test = np.full(len(y_test), prior)
naive_costs = naive_policy_cost(y_test)
log(f"Baseline: always predict P(hit)={prior:.3f}; cheapest no-model policy on test = "
    f"'{min(naive_costs, key=naive_costs.get)}' (${min(naive_costs.values()):,.0f})")

# ---------------------------------------------------------------------------
# 5a. scikit-learn (also picks C, which all three share)
# ---------------------------------------------------------------------------
def fit_sklearn(C: float) -> LogisticRegression:
    return LogisticRegression(C=C, solver="lbfgs", max_iter=5000, tol=1e-8).fit(X_train, y_train)


C_grid = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
val_loss_by_C = {}
for C in C_grid:
    m = fit_sklearn(C)
    val_loss_by_C[C] = summary_metrics(y_val, m.predict_proba(X_val)[:, 1], 0.5)["log_loss"]
    log(f"  sklearn  C={C:<6} val log-loss={val_loss_by_C[C]:.4f}")
best_C = min(val_loss_by_C, key=val_loss_by_C.get)
sk_model = fit_sklearn(best_C)
lam = l2_strength(best_C, len(y_train))
log(f"Chosen C={best_C} (same L2 penalty in PyTorch = {lam:.2e} * ||w||^2)")

Xtr_t = torch.tensor(X_train)
ytr_t = torch.tensor(y_train, dtype=torch.float32)
Xva_t = torch.tensor(X_val)
yva_t = torch.tensor(y_val, dtype=torch.float32)
n_features = X_train.shape[1]


# ---------------------------------------------------------------------------
# 5b. Manual PyTorch loop: raw tensors, autograd, hand-written update
# ---------------------------------------------------------------------------
def train_manual(lr: float, epochs: int = 3000):
    w = torch.zeros(n_features, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    history = []
    for epoch in range(epochs):
        logits = Xtr_t @ w + b               # forward pass (all rows at once)
        loss = bce_with_l2(logits, ytr_t, w, lam)
        loss.backward()                      # autograd fills w.grad and b.grad
        with torch.no_grad():                # the manual gradient-descent step
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                       # gradients add up otherwise
        b.grad.zero_()
        if epoch % 25 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                v = bce_with_l2(Xva_t @ w + b, yva_t, w, lam)
            history.append({"epoch": epoch, "train": loss.item(), "val": v.item()})
    return {"w": w.detach(), "b": b.detach()}, history


# ---------------------------------------------------------------------------
# 5c. Standard workflow: nn.Module + DataLoader + torch.optim
# ---------------------------------------------------------------------------
def train_nn(lr: float, epochs: int = 300, batch_size: int = 256):
    torch.manual_seed(SEED)
    net = LogisticRegressionNet(n_features)
    nn.init.zeros_(net.linear.weight)        # same starting point as the manual loop
    nn.init.zeros_(net.linear.bias)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Mini-batches make every step a bit noisy; shrinking the learning rate
    # towards 0 by the end lets Adam settle ON the minimum, not around it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    history = []
    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = bce_with_l2(net(xb), yb, net.linear.weight, lam)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 3 == 0 or epoch == epochs - 1:
            net.eval()
            with torch.no_grad():
                tr = bce_with_l2(net(Xtr_t), ytr_t, net.linear.weight, lam)
                va = bce_with_l2(net(Xva_t), yva_t, net.linear.weight, lam)
            history.append({"epoch": epoch, "train": float(tr), "val": float(va)})
    net.eval()
    return net, history


# Learning rate: the only other hyper-parameter I tune, picked on val loss.
manual_runs, nn_runs = {}, {}
for lr in [0.05, 0.2, 0.5, 1.0]:
    t0 = time.time()
    manual_runs[lr] = train_manual(lr)
    h = manual_runs[lr][1]
    log(f"  manual    lr={lr:<5} train obj @epoch~300={h[12]['train']:.6f}  final train obj={h[-1]['train']:.6f}  val={h[-1]['val']:.5f}  ({time.time() - t0:.1f}s)")
for lr in [0.001, 0.01, 0.05]:
    t0 = time.time()
    nn_runs[lr] = train_nn(lr)
    h = nn_runs[lr][1]
    log(f"  nn.Module lr={lr:<5} train obj @epoch~30={h[10]['train']:.6f}  final train obj={h[-1]['train']:.6f}  val={h[-1]['val']:.5f}  ({time.time() - t0:.1f}s)")
# How I pick the learning rate: by the TRAINING objective each run reaches,
# not by validation loss. The learning rate is an optimiser setting: its
# only job is to get to the minimum of the objective that C already defines
# (C is the model setting, and C is chosen on validation). When I first
# picked it on validation loss, the winner was a run that had NOT finished
# converging - stopping early acts like extra regularisation, so validation
# was quietly rewarding an unfinished model.
best_lr_manual = min(manual_runs, key=lambda k: (manual_runs[k][1][-1]["train"], k))
best_lr_nn = min(nn_runs, key=lambda k: (nn_runs[k][1][-1]["train"], k))
manual_model, manual_hist = manual_runs[best_lr_manual]
nn_model, nn_hist = nn_runs[best_lr_nn]
log(f"Chosen learning rates: manual={best_lr_manual}, nn.Module={best_lr_nn}")

models = {"scikit-learn": sk_model, "Manual PyTorch": manual_model, "nn.Module PyTorch": nn_model}

# ---------------------------------------------------------------------------
# 6. Evaluate: threshold chosen on VAL (business cost), metrics on TEST
# ---------------------------------------------------------------------------
results, thresholds, test_probs = {}, {}, {}
for name in MODEL_NAMES:
    p_val = predict_proba(name, models[name], X_val)
    thresholds[name] = best_threshold(y_val, p_val)
    test_probs[name] = predict_proba(name, models[name], X_test)
    results[name] = summary_metrics(y_test, test_probs[name], thresholds[name])
# Apples-to-apples: every model at the SAME threshold (sklearn's). The
# validation cost curve is flat near its minimum, so tiny probability
# differences can move each model's own "best" threshold by a step or two.
common_t = thresholds["scikit-learn"]
for name in MODEL_NAMES:
    results[name]["cost_at_common_threshold"] = business_cost(y_test, test_probs[name], common_t)
baseline_name = "Naive baseline"
results[baseline_name] = summary_metrics(y_test, p_baseline_test, 0.5)
results[baseline_name]["business_cost"] = min(naive_costs.values())
results[baseline_name]["cost_at_common_threshold"] = min(naive_costs.values())

# Do the three implementations agree?
W = {name: coefficients(name, models[name]) for name in MODEL_NAMES}
agreement = {}
for name in MODEL_NAMES[1:]:
    agreement[name] = {
        "max_abs_prob_diff_vs_sklearn": float(np.max(np.abs(test_probs[name] - test_probs["scikit-learn"]))),
        "max_abs_coef_diff_vs_sklearn": float(np.max(np.abs(W[name][0] - W["scikit-learn"][0]))),
        "intercept_diff_vs_sklearn": float(W[name][1] - W["scikit-learn"][1]),
        "coef_correlation_vs_sklearn": float(np.corrcoef(W[name][0], W["scikit-learn"][0])[0, 1]),
    }


def objective(name: str) -> float:
    """Value of the shared training objective each model reached."""
    p = np.clip(predict_proba(name, models[name], X_train), 1e-12, 1 - 1e-12)
    bce = -np.mean(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
    return float(bce + lam * np.sum(W[name][0] ** 2))


for name in MODEL_NAMES:
    results[name]["train_objective"] = objective(name)

table = pd.DataFrame(results).T[["roc_auc", "pr_auc", "log_loss", "brier", "threshold",
                                 "precision", "recall", "business_cost",
                                 "cost_at_common_threshold"]]
log("\nTEST SET RESULTS\n" + table.round(4).to_string())
log("\nAGREEMENT VS SCIKIT-LEARN\n"
    + pd.DataFrame(agreement).T.to_string(float_format=lambda v: f"{v:.2e}"))
log("\nTraining objective reached: "
    + ", ".join(f"{k}={results[k]['train_objective']:.6f}" for k in MODEL_NAMES))


# ---------------------------------------------------------------------------
# 6b. Does the feature engineering actually help? (validation set, sklearn)
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score  # noqa: E402

groups = {
    "brand stats (hit rate, size, unseen)": ["brand_hit_rate", "brand_log_n_products", "brand_unseen"],
    "relative price (vs sub-category / brand)": ["price_vs_subcategory", "unit_price_vs_subcategory", "price_vs_brand"],
    "categories": [f for f in feature_names if f.startswith(("cat=", "subcat="))],
    "variants / shades": ["log_child_count", "variant_price_spread"] + [f for f in feature_names if f.startswith("variation=")],
    "name keywords + ingredients": ["name_is_set", "name_is_mini", "name_is_refill", "log_n_ingredients", "ingredients_missing"],
}
full_val_auc = roc_auc_score(y_val, sk_model.predict_proba(X_val)[:, 1])
ablation = {"all features": full_val_auc}
for group, cols in groups.items():
    keep_idx = [i for i, f in enumerate(feature_names) if f not in cols]
    m = LogisticRegression(C=best_C, max_iter=5000).fit(X_train[:, keep_idx], y_train)
    ablation[f"without {group}"] = roc_auc_score(y_val, m.predict_proba(X_val[:, keep_idx])[:, 1])
simple = ["log_price", "limited_edition", "online_only", "sephora_exclusive"] + groups["categories"]
keep_idx = [feature_names.index(f) for f in simple]
m = LogisticRegression(C=best_C, max_iter=5000).fit(X_train[:, keep_idx], y_train)
ablation["only raw columns (price, flags, category)"] = roc_auc_score(y_val, m.predict_proba(X_val[:, keep_idx])[:, 1])
# The highlight tags I built and then rejected: add them back to show why.
hl_builder = FeatureBuilder(use_highlights=True).fit(train, train["hit"])
m = LogisticRegression(C=best_C, max_iter=5000).fit(hl_builder.transform(train, training_rows=True), y_train)
ablation["WITH highlight tags added back (rejected)"] = roc_auc_score(
    y_val, m.predict_proba(hl_builder.transform(val))[:, 1])
log("\nFEATURE ABLATION (validation ROC-AUC, sklearn, same C)")
for k, v in ablation.items():
    log(f"  {k:<48} {v:.4f}  ({v - full_val_auc:+.4f})")

# What if I had split at random instead of by listing time? (diagnostic only)
shuffled = df.sample(frac=1.0, random_state=SEED)
r_train, r_test = shuffled.iloc[:cut_train], shuffled.iloc[cut_val:]
r_builder = FeatureBuilder().fit(r_train, r_train["hit"])
r_model = LogisticRegression(C=best_C, max_iter=5000).fit(
    r_builder.transform(r_train, training_rows=True), r_train["hit"])
random_split_auc = roc_auc_score(r_test["hit"], r_model.predict_proba(r_builder.transform(r_test))[:, 1])
log(f"Random-split test AUC would be {random_split_auc:.4f} vs time-split {results['scikit-learn']['roc_auc']:.4f}")

# ---------------------------------------------------------------------------
# 7. Save everything (the dashboard only LOADS these)
# ---------------------------------------------------------------------------
joblib.dump(sk_model, MODELS_DIR / "sklearn_logreg.joblib")
torch.save(manual_model, MODELS_DIR / "manual_torch_logreg.pt")
torch.save({"state_dict": nn_model.state_dict(), "n_features": n_features},
           MODELS_DIR / "nn_torch_logreg.pt")
joblib.dump(builder, ARTIFACTS_DIR / "feature_builder.joblib")

# Which product went to which split, with its label. The app re-reads the raw
# CSV and runs the fitted FeatureBuilder + saved models on the test rows.
df[["product_id", "split", "hit", "listing_rank"]].to_csv(ARTIFACTS_DIR / "splits.csv", index=False)

# Readable (unscaled) features for the distribution plots
raw_feats = pd.concat([builder.transform_raw(train, training_rows=True),
                       builder.transform_raw(val), builder.transform_raw(test)])
raw_feats.insert(0, "product_id", df.loc[raw_feats.index, "product_id"])
raw_feats.to_csv(ARTIFACTS_DIR / "engineered_features.csv", index=False)

coef_df = pd.DataFrame({name: W[name][0] for name in MODEL_NAMES}, index=feature_names)
coef_df.loc["(intercept)"] = [W[name][1] for name in MODEL_NAMES]
coef_df.to_csv(ARTIFACTS_DIR / "coefficients.csv", index_label="feature")

summary = {
    "chosen_C": best_C, "l2_strength": lam, "val_log_loss_by_C": val_loss_by_C,
    "lr_manual": best_lr_manual, "lr_nn": best_lr_nn,
    "final_train_objective_by_lr_manual": {k: v[1][-1]["train"] for k, v in manual_runs.items()},
    "final_train_objective_by_lr_nn": {k: v[1][-1]["train"] for k, v in nn_runs.items()},
    "val_loss_by_lr_manual": {k: v[1][-1]["val"] for k, v in manual_runs.items()},
    "val_loss_by_lr_nn": {k: v[1][-1]["val"] for k, v in nn_runs.items()},
    "thresholds_from_val": thresholds,
    "cost_false_positive": COST_FALSE_POSITIVE, "cost_false_negative": COST_FALSE_NEGATIVE,
    "naive_policy_costs_test": naive_costs, "train_hit_rate": prior,
    "n_rows": {"train": len(train), "val": len(val), "test": len(test)},
    "test_results": results, "agreement": agreement,
    "loss_curves": {"Manual PyTorch": manual_hist, "nn.Module PyTorch": nn_hist},
    "loss_curves_all_lr": {"Manual PyTorch": {str(k): v[1] for k, v in manual_runs.items()},
                           "nn.Module PyTorch": {str(k): v[1] for k, v in nn_runs.items()}},
    "ablation_val_auc": ablation, "random_split_test_auc": random_split_auc,
    "category_hit_rates": df.groupby(category_group(df))["hit"].mean().to_dict(),
}
(ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
log(f"\nSaved models to {MODELS_DIR.name}/ and dashboard artifacts to {ARTIFACTS_DIR.name}/")
