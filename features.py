"""
Feature engineering for the Sephora "hit product" model.

Shared by train.py (which FITS the FeatureBuilder on the training rows only)
and app.py (which only LOADS the fitted builder and calls .transform()).

Rule I follow everywhere in this file: anything that "learns" from data
(medians, category lists, brand statistics, the scaler) is learned in
.fit() from the TRAINING rows only. Validation and test rows are only ever
passed through .transform(), so no information from them leaks in.
"""

import ast
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Columns that only exist AFTER a product has been on sale for a while.
# A buyer deciding on listing day can't know them, so they are never features.
POST_LAUNCH_COLUMNS = [
    "loves_count", "rating", "reviews", "out_of_stock", "sale_price_usd",
]

SMALL_CATEGORIES = {"Men": "Other", "Tools & Brushes": "Other"}

OZ_TO_ML = 29.5735


# ---------------------------------------------------------------------------
# 1. Cleaning helpers (pure functions, nothing learned)
# ---------------------------------------------------------------------------

def listing_order(df: pd.DataFrame) -> pd.Series:
    """Product IDs (P123456) were handed out over time, so the numeric part
    works as a 'listing date' proxy. A handful of IDs are absurdly large
    (gift-card style codes), so I use the RANK, not the raw number."""
    return df["product_id"].str[1:].astype("int64").rank(method="first")


def category_group(df: pd.DataFrame) -> pd.Series:
    """Primary category with the two tiny ones (Men, Tools) merged."""
    return df["primary_category"].replace(SMALL_CATEGORIES)


def parse_size_amount(size: str) -> float:
    """Turn '3.4 oz/ 100 mL' or '0.31 oz/ 9 g' into one number (mL or g).
    mL and g are treated as roughly the same unit: it's only used to compute
    a price-per-unit that is then compared WITHIN a sub-category."""
    if not isinstance(size, str):
        return np.nan
    s = size.lower().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*ml", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?oz", s)
    if m:
        return float(m.group(1)) * OZ_TO_ML
    return np.nan


def parse_list(text) -> list:
    """highlights / ingredients are stored as the string of a Python list."""
    if not isinstance(text, str):
        return []
    try:
        with warnings.catch_warnings():  # some ingredient strings contain "\A" etc.
            warnings.simplefilter("ignore", SyntaxWarning)
            value = ast.literal_eval(text)
        return value if isinstance(value, list) else []
    except (ValueError, SyntaxError):
        return []


def count_ingredients(text) -> float:
    """Rough ingredient count: split every entry of the list on commas."""
    items = parse_list(text)
    if not items:
        return np.nan
    return float(sum(len([p for p in str(i).split(",") if p.strip()]) for i in items))


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Listing-day information only, cleaned into simple numeric columns.
    Nothing here depends on other rows, so it is safe to run on any split."""
    out = pd.DataFrame(index=df.index)
    name = df["product_name"].str.lower().fillna("")

    out["price_usd"] = df["price_usd"]
    out["log_price"] = np.log(df["price_usd"].clip(lower=1))
    amount = df["size"].map(parse_size_amount)
    out["size_missing"] = amount.isna().astype(float)
    out["log_unit_price"] = np.log(df["price_usd"].clip(lower=1) / amount.clip(lower=0.1))

    out["log_child_count"] = np.log1p(df["child_count"])
    spread = (df["child_max_price"] - df["child_min_price"]).fillna(0)
    out["variant_price_spread"] = (spread / df["price_usd"].clip(lower=1)).clip(0, 5)

    out["has_value_price"] = df["value_price_usd"].notna().astype(float)
    out["value_premium"] = (
        (df["value_price_usd"] / df["price_usd"] - 1).fillna(0).clip(0, 5)
    )

    for flag in ["limited_edition", "online_only", "sephora_exclusive"]:
        out[flag] = df[flag].astype(float)

    n_ing = df["ingredients"].map(count_ingredients)
    out["ingredients_missing"] = n_ing.isna().astype(float)
    out["log_n_ingredients"] = np.log1p(n_ing)

    highlights = df["highlights"].map(parse_list)
    out["n_highlights"] = highlights.map(len).astype(float)

    out["name_is_set"] = name.str.contains(r"\b(?:set|kit|duo|trio|collection|bundle)\b").astype(float)
    out["name_is_mini"] = name.str.contains(r"\b(?:mini|travel)\b").astype(float)
    out["name_is_refill"] = name.str.contains(r"\brefill\b").astype(float)

    # Kept as text, turned into one-hot columns by the fitted builder.
    out["_brand"] = df["brand_name"]
    out["_category"] = category_group(df)
    out["_subcategory"] = df["secondary_category"].fillna("Unknown")
    out["_variation"] = df["variation_type"].fillna("None")
    out["_highlights"] = highlights
    return out


# ---------------------------------------------------------------------------
# 2. Target
# ---------------------------------------------------------------------------

def make_target(df: pd.DataFrame, n_cohorts: int = 10, quantile: float = 0.75) -> pd.Series:
    """'Hit' = in the top 25% of loves among products of the SAME category
    listed in the SAME period (one of 10 equal-size listing-order cohorts).

    Why not just 'top 25% of loves overall'? Because loves pile up over time:
    the oldest tenth of products has ~7x the median loves of the newest
    tenth. A raw threshold would mostly learn 'is this product old', which is
    useless for a product that is being listed today."""
    cohort = pd.qcut(listing_order(df), n_cohorts, labels=False)
    group = [cohort, category_group(df)]
    cutoff = df.groupby(group)["loves_count"].transform(lambda s: s.quantile(quantile))
    return (df["loves_count"] >= cutoff).astype(int)


# ---------------------------------------------------------------------------
# 3. The fitted part
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Learns everything data-dependent from the TRAINING rows only."""

    def __init__(self, use_highlights: bool = False, n_top_highlights: int = 20,
                 min_subcat_rows: int = 30, brand_smoothing: float = 20.0,
                 random_state: int = 42):
        # Highlight tags are OFF by default: the ablation in train.py showed
        # they hurt on newer products because Sephora's tagging changed over
        # time (concept drift). See REPORT.md.
        self.use_highlights = use_highlights
        self.n_top_highlights = n_top_highlights
        self.min_subcat_rows = min_subcat_rows
        self.brand_smoothing = brand_smoothing
        self.random_state = random_state

    # -- brand statistics ---------------------------------------------------
    def _brand_tables(self, brands: pd.Series, y: pd.Series, scale: float = 1.0) -> dict:
        """Per-brand portfolio size and smoothed hit rate:
        rate = (hits + m*prior) / (n + m). Brands with few products get
        pulled towards the overall rate so one lucky product doesn't make a
        brand look like a hit factory."""
        stats = pd.DataFrame({"brand": brands, "y": y}).groupby("brand")["y"].agg(["sum", "count"])
        m = self.brand_smoothing
        return {"n": stats["count"] * scale,
                "rate": (stats["sum"] + m * self.prior_) / (stats["count"] + m)}

    def _apply_brand_tables(self, brands: pd.Series, tables: dict) -> pd.DataFrame:
        n = brands.map(tables["n"])
        return pd.DataFrame({
            "brand_unseen": n.isna().astype(float),
            "brand_log_n_products": np.log1p(n.fillna(0)),
            "brand_hit_rate": brands.map(tables["rate"]).fillna(self.prior_),
        }, index=brands.index)

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "FeatureBuilder":
        b = base_frame(df)
        y = pd.Series(np.asarray(y), index=df.index)
        self.prior_ = float(y.mean())

        # Category lists come from training data; unseen values -> "Other".
        self.categories_ = sorted(b["_category"].unique())
        counts = b["_subcategory"].value_counts()
        self.subcategories_ = sorted(counts[counts >= self.min_subcat_rows].index)
        self.variations_ = ["Size", "Color", "Size + Concentration + Formulation", "None"]
        tag_counts = pd.Series([t for tags in b["_highlights"] for t in tags]).value_counts()
        self.top_highlights_ = list(tag_counts.head(self.n_top_highlights).index)

        # "Is this expensive FOR ITS KIND of product?" reference points.
        self.subcat_log_price_ = b.groupby("_subcategory")["log_price"].median()
        self.subcat_unit_price_ = b.groupby("_subcategory")["log_unit_price"].median()
        self.global_log_price_ = float(b["log_price"].median())
        self.global_unit_price_ = float(b["log_unit_price"].median())
        self.ingredient_median_ = float(b["log_n_ingredients"].median())
        self.brand_log_price_ = b.groupby("_brand")["log_price"].median()

        # Brand statistics used for validation/test/new products: all of train.
        self.brand_tables_ = self._brand_tables(b["_brand"], y)

        # Out-of-fold brand statistics for the training rows themselves: each
        # row's brand stats are computed from the OTHER 4 folds. Two reasons:
        # (1) the row's own label never hides inside its brand hit rate, and
        # (2) brands that only appear in one fold look "unseen", so the model
        # learns how to treat a brand it has never seen before, which happens
        # a lot in the newest products (new brands join Sephora all the time).
        parts = []
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for fit_idx, enc_idx in kf.split(df):
            tables = self._brand_tables(b["_brand"].iloc[fit_idx], y.iloc[fit_idx], scale=5 / 4)
            parts.append(self._apply_brand_tables(b["_brand"].iloc[enc_idx], tables))
        self._train_oof_brand_ = pd.concat(parts).loc[df.index]

        X = self._assemble(b, brand=self._train_oof_brand_)
        self.feature_names_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X.values)
        return self

    def _assemble(self, b: pd.DataFrame, brand: pd.DataFrame | None = None) -> pd.DataFrame:
        X = pd.DataFrame(index=b.index)
        sub_price = b["_subcategory"].map(self.subcat_log_price_).fillna(self.global_log_price_)
        sub_unit = b["_subcategory"].map(self.subcat_unit_price_).fillna(self.global_unit_price_)
        unit = b["log_unit_price"].fillna(sub_unit)  # missing size -> typical for its kind

        X["log_price"] = b["log_price"]
        X["price_vs_subcategory"] = b["log_price"] - sub_price
        X["unit_price_vs_subcategory"] = unit - sub_unit
        for col in ["size_missing", "log_child_count", "variant_price_spread",
                    "has_value_price", "value_premium", "limited_edition",
                    "online_only", "sephora_exclusive", "ingredients_missing",
                    "name_is_set", "name_is_mini", "name_is_refill"]:
            X[col] = b[col]
        X["log_n_ingredients"] = b["log_n_ingredients"].fillna(self.ingredient_median_)

        if brand is None:
            brand = self._apply_brand_tables(b["_brand"], self.brand_tables_)
        for col in ["brand_unseen", "brand_log_n_products", "brand_hit_rate"]:
            X[col] = brand[col]
        X["price_vs_brand"] = b["log_price"] - b["_brand"].map(self.brand_log_price_).fillna(b["log_price"])

        for c in self.categories_:
            X[f"cat={c}"] = (b["_category"] == c).astype(float)
        sub = b["_subcategory"].where(b["_subcategory"].isin(self.subcategories_), "Other")
        for c in self.subcategories_ + ["Other"]:
            X[f"subcat={c}"] = (sub == c).astype(float)
        var = b["_variation"].where(b["_variation"].isin(self.variations_), "Other")
        for v in self.variations_ + ["Other"]:
            X[f"variation={v}"] = (var == v).astype(float)
        if self.use_highlights:
            X["n_highlights"] = b["n_highlights"]
            for t in self.top_highlights_:
                X[f"tag={t}"] = b["_highlights"].map(lambda tags, t=t: float(t in tags))
        return X

    # -- public API ---------------------------------------------------------
    def transform_raw(self, df: pd.DataFrame, training_rows: bool = False) -> pd.DataFrame:
        """Unscaled features (human-readable, used for the dashboard plots)."""
        b = base_frame(df)
        brand = self._train_oof_brand_.loc[df.index] if training_rows else None
        return self._assemble(b, brand=brand)[self.feature_names_]

    def transform(self, df: pd.DataFrame, training_rows: bool = False) -> np.ndarray:
        """Scaled feature matrix, the exact input all three models receive."""
        X = self.transform_raw(df, training_rows=training_rows)
        return self.scaler_.transform(X.values).astype(np.float32)
