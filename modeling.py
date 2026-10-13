"""
Model definitions and evaluation helpers shared by train.py and app.py.

All three implementations fit the SAME model (logistic regression) with the
SAME objective, so if training works they should land on (almost) the same
weights:

    loss(w, b) = mean binary cross-entropy  +  (1 / (2 * C * n)) * ||w||^2

That second term is exactly scikit-learn's L2 penalty rewritten for a MEAN
loss: sklearn minimises  C * sum(BCE) + 0.5 * ||w||^2 ; dividing everything
by C*n gives the formula above. The intercept b is not penalised, same as
sklearn.
"""

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

MODEL_NAMES = ["scikit-learn", "Manual PyTorch", "nn.Module PyTorch"]

# Business costs (USD) for one product decision. Assumptions, explained in
# REPORT.md, and adjustable in the dashboard.
COST_FALSE_POSITIVE = 300.0   # promoted a product that did NOT become a hit
COST_FALSE_NEGATIVE = 1000.0  # did NOT promote a product that became a hit


def l2_strength(C: float, n: int) -> float:
    """Coefficient in front of ||w||^2 that matches sklearn's C."""
    return 1.0 / (2.0 * C * n)


def bce_with_l2(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, lam: float) -> torch.Tensor:
    """The shared objective. Uses the numerically stable logits version of
    binary cross-entropy instead of sigmoid() followed by log()."""
    return nn.functional.binary_cross_entropy_with_logits(logits, y) + lam * (w ** 2).sum()


class LogisticRegressionNet(nn.Module):
    """Standard-workflow model: one linear layer = w.x + b. The sigmoid is
    applied inside the loss (training) or in predict_proba (inference)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)  # logits, shape (n,)


# ---------------------------------------------------------------------------
# Loading saved models + one predict function for all three
# ---------------------------------------------------------------------------

def load_models() -> dict:
    """Load the three trained models from disk (never retrains anything)."""
    sk = joblib.load(MODELS_DIR / "sklearn_logreg.joblib")
    manual = torch.load(MODELS_DIR / "manual_torch_logreg.pt", weights_only=True)
    ckpt = torch.load(MODELS_DIR / "nn_torch_logreg.pt", weights_only=True)
    net = LogisticRegressionNet(ckpt["n_features"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return {"scikit-learn": sk, "Manual PyTorch": manual, "nn.Module PyTorch": net}


def predict_proba(name: str, model, X: np.ndarray) -> np.ndarray:
    """P(hit) for each row of the (already scaled) feature matrix X."""
    if name == "scikit-learn":
        return model.predict_proba(X)[:, 1]
    Xt = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        if name == "Manual PyTorch":
            logits = Xt @ model["w"] + model["b"]
        else:
            logits = model(Xt)
    return torch.sigmoid(logits).numpy()


def coefficients(name: str, model) -> tuple[np.ndarray, float]:
    """(weights, intercept) in the same layout for all three models."""
    if name == "scikit-learn":
        return model.coef_.ravel(), float(model.intercept_[0])
    if name == "Manual PyTorch":
        return model["w"].numpy().ravel(), float(model["b"])
    return model.linear.weight.detach().numpy().ravel(), float(model.linear.bias.detach())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    return {
        "tp": int(np.sum(pred & (y == 1))), "fp": int(np.sum(pred & (y == 0))),
        "fn": int(np.sum(~pred & (y == 1))), "tn": int(np.sum(~pred & (y == 0))),
    }


def business_cost(y, p, threshold, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE) -> float:
    cm = confusion_at(np.asarray(y), np.asarray(p), threshold)
    return cm["fp"] * cost_fp + cm["fn"] * cost_fn


def best_threshold(y, p, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE) -> float:
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    costs = [business_cost(y, p, t, cost_fp, cost_fn) for t in grid]
    return float(grid[int(np.argmin(costs))])


def naive_policy_cost(y, cost_fp=COST_FALSE_POSITIVE, cost_fn=COST_FALSE_NEGATIVE) -> dict:
    """The two no-model policies a buyer could follow."""
    y = np.asarray(y)
    return {"promote nothing": float(y.sum() * cost_fn),
            "promote everything": float((1 - y).sum() * cost_fp)}


def summary_metrics(y, p, threshold) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    cm = confusion_at(y, p, threshold)
    precision = cm["tp"] / max(cm["tp"] + cm["fp"], 1)
    recall = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    return {
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "business_cost": float(business_cost(y, p, threshold)),
        **cm,
    }
