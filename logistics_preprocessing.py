"""
logistics_preprocessing.py
==========================
End-to-end data cleaning & preprocessing pipeline for a shipment-level
logistics dataset.

Pipeline
--------
 1. Load            read everything as text, treat placeholder tokens as missing
 2. Audit (raw)     rows, duplicates, missing values per column
 3. Standardise     text labels (cities, carriers, status), numbers, dates
 4. De-duplicate    one record per shipment_id (keep the most complete one)
 5. Validate        business rules -> impossible values become missing
 6. Impute          domain-aware imputation (route / vehicle / expected transit)
 7. Outliers        weight: IQR on log scale per vehicle (cap); cost: regression residual + MAD (correct);
                   delays: IQR among late shipments (flag only, never removed)
 8. Features        transit_days, delay_days, is_late, cost_per_km ...
 9. Normalise       log1p + Min-Max (skewed) and Z-score (transit / delay)
10. Export          cleaned CSV, quality report (JSON) and figures

Usage
-----
    python logistics_preprocessing.py
    python logistics_preprocessing.py --input logistics_raw.csv --output logistics_cleaned.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --------------------------------------------------------------------------
# Configuration (domain knowledge lives here, not buried in the code)
# --------------------------------------------------------------------------
MISSING_TOKENS = ["", "NA", "N/A", "n/a", "null", "NULL", "None", "-", "--"]

CITY_ALIASES = {
    "delhi": "Delhi", "new delhi": "Delhi",
    "mumbai": "Mumbai", "bombay": "Mumbai",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
    "kolkata": "Kolkata", "calcutta": "Kolkata",
    "chennai": "Chennai", "madras": "Chennai",
    "hyderabad": "Hyderabad", "patna": "Patna", "pune": "Pune",
    "ahmedabad": "Ahmedabad", "jaipur": "Jaipur", "lucknow": "Lucknow",
}
STATUS_MAP = {"delivered": "Delivered", "in transit": "In Transit", "returned": "Returned"}

VEHICLE_CAPACITY_KG = {"Mini Truck": 2_500, "Truck": 18_000, "Container": 28_000, "Air Cargo": 5_000}
DEFAULT_CAPACITY_KG = 28_000                       # for 'Unknown' vehicle
SPEED_KM_PER_DAY = {"Mini Truck": 350, "Truck": 400, "Container": 450, "Air Cargo": 1_500}
DEFAULT_SPEED = 400

MAX_ROAD_DISTANCE_KM = 4_000                       # longest plausible route within India
MAX_TRANSIT_DAYS = 30                              # anything longer is a data error
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]


# --------------------------------------------------------------------------
# 1-2. Load & audit
# --------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    """Read all columns as text so nothing is silently mis-typed on load."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=MISSING_TOKENS)


def audit(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "duplicate_shipment_ids": int(df["shipment_id"].duplicated().sum()),
        "missing_by_column": {c: int(n) for c, n in df.isna().sum().items()},
    }


# --------------------------------------------------------------------------
# 3. Standardisation
# --------------------------------------------------------------------------
def harmonise_case(s: pd.Series) -> pd.Series:
    """'safehaul', 'SAFEHAUL', ' SafeHaul ' -> the most frequent spelling ('SafeHaul')."""
    s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    known = s.dropna()
    canonical = known.groupby(known.str.lower()).agg(lambda x: x.value_counts().idxmax())
    return s.str.lower().map(canonical)


def to_number(s: pd.Series) -> pd.Series:
    """'₹12,450' / 'INR 12,450' / '1250.5 kg' -> float (keeps minus sign)."""
    return pd.to_numeric(s.str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def parse_dates(s: pd.Series) -> pd.Series:
    """Try each known format explicitly - never let pandas guess day/month order."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["origin_city", "destination_city"]:
        key = df[col].str.strip().str.lower()
        df[col] = key.map(CITY_ALIASES).fillna(key.str.title())
    for col in ["carrier", "vehicle_type", "priority"]:
        df[col] = harmonise_case(df[col])
    status_key = df["delivery_status"].str.strip().str.lower().str.replace("-", " ", regex=False)
    df["delivery_status"] = status_key.map(STATUS_MAP)
    for col in ["order_date", "dispatch_date", "delivery_date"]:
        df[col] = parse_dates(df[col])
    for col in ["distance_km", "weight_kg", "freight_cost_inr"]:
        df[col] = to_number(df[col])
    return df


# --------------------------------------------------------------------------
# 4. De-duplication
# --------------------------------------------------------------------------
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep, for every shipment_id, the record with the fewest missing fields."""
    return (df.assign(_n_missing=df.isna().sum(axis=1))
              .sort_values(["shipment_id", "_n_missing"])
              .drop_duplicates("shipment_id", keep="first")
              .drop(columns="_n_missing")
              .sort_values("shipment_id").reset_index(drop=True))


# --------------------------------------------------------------------------
# 5. Business-rule validation (impossible values -> NaN)
# --------------------------------------------------------------------------
def apply_business_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    found = {}

    bad = (df["distance_km"] <= 0) | (df["distance_km"] > MAX_ROAD_DISTANCE_KM)
    found["invalid_distance"] = int(bad.sum()); df.loc[bad, "distance_km"] = np.nan

    capacity = df["vehicle_type"].map(VEHICLE_CAPACITY_KG).fillna(DEFAULT_CAPACITY_KG)
    bad = (df["weight_kg"] <= 0) | (df["weight_kg"] > capacity)
    found["weight_exceeds_vehicle_capacity"] = int(bad.sum()); df.loc[bad, "weight_kg"] = np.nan

    bad = df["freight_cost_inr"] <= 0
    found["non_positive_cost"] = int(bad.sum()); df.loc[bad, "freight_cost_inr"] = np.nan

    transit = (df["delivery_date"] - df["dispatch_date"]).dt.days
    bad = (transit < 0) | (transit > MAX_TRANSIT_DAYS)
    found["impossible_delivery_date"] = int(bad.sum()); df.loc[bad, "delivery_date"] = pd.NaT

    found["dispatch_before_order"] = int((df["dispatch_date"] < df["order_date"]).sum())
    return df, found


# --------------------------------------------------------------------------
# 6. Missing-value imputation
# --------------------------------------------------------------------------
def impute(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    before = {c: int(df[c].isna().sum()) for c in
              ["carrier", "vehicle_type", "distance_km", "weight_kg", "freight_cost_inr", "delivery_date"]}

    # categorical: explicit 'Unknown' keeps the information that the value was absent
    df["carrier"] = df["carrier"].fillna("Unknown")
    df["vehicle_type"] = df["vehicle_type"].fillna("Unknown")

    # distance: median of the same origin->destination route
    df["distance_imputed"] = df["distance_km"].isna()
    route_median = df.groupby(["origin_city", "destination_city"])["distance_km"].transform("median")
    df["distance_km"] = df["distance_km"].fillna(route_median).fillna(df["distance_km"].median())

    # weight: median of the same vehicle type (a Mini Truck never carries 15 t)
    df["weight_imputed"] = df["weight_kg"].isna()
    df["weight_kg"] = df["weight_kg"].fillna(df.groupby("vehicle_type")["weight_kg"].transform("median"))

    # freight cost: median cost-per-km of the same vehicle type and distance band x distance
    df["cost_imputed"] = df["freight_cost_inr"].isna()
    band = pd.cut(df["distance_km"], [0, 500, 1_000, 1_500, 2_500, np.inf]).astype(str)
    cpk = df["freight_cost_inr"] / df["distance_km"]
    cpk_med = cpk.groupby([df["vehicle_type"], band]).transform("median")
    cpk_med = cpk_med.fillna(cpk.groupby(df["vehicle_type"]).transform("median")).fillna(cpk.median())
    df["freight_cost_inr"] = df["freight_cost_inr"].fillna((cpk_med * df["distance_km"]).round(0))

    # delivery date: only for shipments that SHOULD have one. 'In Transit' stays empty on purpose.
    expected = expected_transit(df)
    observed = (df["delivery_date"] - df["dispatch_date"]).dt.days
    typical_delay = float((observed - expected).median())
    need = df["delivery_date"].isna() & df["delivery_status"].isin(["Delivered", "Returned"])
    df["delivery_imputed"] = need
    fill = df.loc[need, "dispatch_date"] + pd.to_timedelta(expected[need] + round(typical_delay), unit="D")
    df.loc[need, "delivery_date"] = fill

    return df, {"missing_before_imputation": before, "typical_delay_days": typical_delay}


def expected_transit(df: pd.DataFrame) -> pd.Series:
    speed = df["vehicle_type"].map(SPEED_KM_PER_DAY).fillna(DEFAULT_SPEED)
    return np.ceil(df["distance_km"] / speed).clip(lower=1)


# --------------------------------------------------------------------------
# 7. Outlier handling
# --------------------------------------------------------------------------
def iqr_bounds(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def cap_outliers(df: pd.DataFrame, col: str, group: str, k: float = 1.5) -> pd.DataFrame:
    """Winsorise on the log scale per group; adds <col>_outlier flag."""
    df = df.copy()
    logged = np.log(df[col])
    flag = pd.Series(False, index=df.index)
    for _, idx in df.groupby(group).groups.items():
        lo, hi = iqr_bounds(logged.loc[idx], k)
        flag.loc[idx] = (logged.loc[idx] < lo) | (logged.loc[idx] > hi)
        logged.loc[idx] = logged.loc[idx].clip(lo, hi)
    df[f"{col}_outlier"] = flag
    df[col] = np.exp(logged).round(1)
    return df


def correct_cost_outliers(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    """
    Freight cost depends on distance, weight and vehicle, so a plain IQR on cost
    would flag every long-haul shipment.  Instead: fit log(cost) ~ log(distance)
    + log(weight) + vehicle, and flag shipments whose residual is extreme
    (modified z-score based on MAD > 3.5).  Flagged costs are replaced by the
    model's prediction (fit again without the flagged rows).
    """
    df = df.copy()
    X = pd.get_dummies(df["vehicle_type"], dtype=float)
    X["log_dist"], X["log_wt"] = np.log(df["distance_km"]), np.log(df["weight_kg"])
    y = np.log(df["freight_cost_inr"])

    def fit(mask):
        beta, *_ = np.linalg.lstsq(X[mask].to_numpy(), y[mask].to_numpy(), rcond=None)
        return X.to_numpy() @ beta

    keep = pd.Series(True, index=df.index)
    for _ in range(2):                                   # 2 passes: robust to the outliers themselves
        resid = y - fit(keep)
        mad = (resid[keep] - resid[keep].median()).abs().median()
        z = 0.6745 * (resid - resid[keep].median()) / mad
        keep = (z.abs() <= z_thresh) | (df["vehicle_type"] == "Unknown")   # model is meaningless for mixed 'Unknown'
    df["cost_outlier"] = ~keep
    df.loc[~keep, "freight_cost_inr"] = np.exp(fit(keep))[~keep].round(0)
    return df


# --------------------------------------------------------------------------
# 8. Feature engineering
# --------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dispatch_lag_days"] = (df["dispatch_date"] - df["order_date"]).dt.days
    df["transit_days"] = (df["delivery_date"] - df["dispatch_date"]).dt.days
    df["expected_transit_days"] = expected_transit(df)
    df["delay_days"] = df["transit_days"] - df["expected_transit_days"]
    df["is_late"] = np.where(df["delay_days"].isna(), np.nan, (df["delay_days"] > 0).astype(float))
    df["cost_per_km"] = (df["freight_cost_inr"] / df["distance_km"]).round(2)
    df["order_month"] = df["order_date"].dt.month

    late = df.loc[df["delay_days"] > 0, "delay_days"]              # IQR among LATE shipments only
    _, hi = iqr_bounds(late, k=1.5)                                # (most shipments have 0 delay -> IQR = 0)
    df["severe_delay_flag"] = df["delay_days"] > hi                # FLAG only - genuine delays are signal
    return df


# --------------------------------------------------------------------------
# 9. Normalisation
# --------------------------------------------------------------------------
def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["weight_kg", "distance_km", "freight_cost_inr"]:      # right-skewed -> log then Min-Max
        df[f"{col}_norm"] = MinMaxScaler().fit_transform(np.log1p(df[[col]]))[:, 0]
    for col in ["transit_days", "delay_days"]:                        # roughly symmetric -> Z-score
        df[f"{col}_z"] = StandardScaler().fit_transform(df[[col]])[:, 0]   # NaN (in transit) is preserved
    return df


# --------------------------------------------------------------------------
# 10. Figures
# --------------------------------------------------------------------------
def make_figures(raw_num: pd.DataFrame, clean: pd.DataFrame, missing_raw: dict,
                 missing_clean: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    navy, orange = "#1F3A5F", "#E07A1F"

    # (a) missing values before / after
    cols = [c for c in missing_raw if c != "delivery_date"] + ["delivery_date"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(cols))
    ax.bar(x - .2, [missing_raw[c] for c in cols], .4, label="Raw", color=orange)
    ax.bar(x + .2, [missing_clean.get(c, 0) for c in cols], .4, label="Cleaned", color=navy)
    ax.set_xticks(x, cols, rotation=35, ha="right"); ax.set_ylabel("Missing cells")
    ax.set_title("Missing values before vs after cleaning"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir / "01_missing_values.png", dpi=170); plt.close(fig)

    # (b) outliers: raw (valid-positive only, log axis) vs cleaned
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.4))
    for ax, (col, label) in zip(axes, [("weight_kg", "Weight (kg)"), ("distance_km", "Distance (km)"),
                                       ("freight_cost_inr", "Freight cost (INR)")]):
        r = raw_num[col].dropna(); r = r[r > 0]
        bp = ax.boxplot([r, clean[col]], tick_labels=["Raw", "Cleaned"], patch_artist=True,
                        flierprops={"markersize": 2.5, "alpha": .5})
        for patch, c in zip(bp["boxes"], [orange, navy]): patch.set_facecolor(c); patch.set_alpha(.75)
        ax.set_yscale("log"); ax.set_title(label)
    fig.suptitle("Outlier effect (log scale)", y=1.0)
    fig.tight_layout(); fig.savefig(outdir / "02_outliers_before_after.png", dpi=170); plt.close(fig)

    # (c) normalisation
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))
    axes[0].hist(clean["weight_kg"], bins=40, color=orange); axes[0].set_title("weight_kg (skewed)")
    axes[1].hist(np.log1p(clean["weight_kg"]), bins=40, color="#7A8CA5"); axes[1].set_title("log1p(weight_kg)")
    axes[2].hist(clean["weight_kg_norm"], bins=40, color=navy); axes[2].set_title("Min-Max scaled [0,1]")
    fig.tight_layout(); fig.savefig(outdir / "03_normalisation.png", dpi=170); plt.close(fig)

    # (d) why quality matters: late-delivery rate by carrier
    d = clean.dropna(subset=["is_late"]).groupby("carrier")["is_late"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.barh(d.index, d.values, color=navy); ax.set_xlabel("% shipments delivered late")
    ax.set_title("Late-delivery rate by carrier (cleaned data)")
    fig.tight_layout(); fig.savefig(outdir / "04_late_by_carrier.png", dpi=170); plt.close(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(input_path: str, output_path: str, report_dir: str = "reports", fig_dir: str = "figures") -> pd.DataFrame:
    report = {}
    raw = load_raw(input_path)
    report["raw"] = audit(raw)

    df = standardise(raw)
    report["unique_labels_after_standardising"] = {
        c: int(df[c].nunique()) for c in ["origin_city", "destination_city", "carrier", "vehicle_type", "delivery_status"]}
    report["unique_labels_before"] = {
        c: int(raw[c].nunique()) for c in ["origin_city", "destination_city", "carrier", "vehicle_type", "delivery_status"]}

    df = drop_duplicates(df)
    report["rows_after_dedup"] = int(len(df))
    raw_numeric = df[["weight_kg", "distance_km", "freight_cost_inr"]].copy()

    df, report["rule_violations"] = apply_business_rules(df)
    df, report["imputation"] = impute(df)

    df = cap_outliers(df, "weight_kg", "vehicle_type")
    report["weight_outliers_capped"] = int(df["weight_kg_outlier"].sum())
    df = correct_cost_outliers(df)
    report["cost_outliers_corrected"] = int(df["cost_outlier"].sum())

    df = add_features(df)
    report["severe_delay_flagged_not_removed"] = int(df["severe_delay_flag"].sum())
    df = normalise(df)

    report["clean"] = audit(df)
    report["in_transit_rows_without_delivery_date"] = int((df["delivery_status"] == "In Transit").sum())
    report["share_of_cells_imputed"] = {
        c: round(float(df[f"{c}_imputed"].mean() * 100), 2)
        for c in ["distance", "weight", "cost", "delivery"]}

    # tidy column order & rounding
    lead = ["shipment_id", "order_date", "dispatch_date", "delivery_date", "origin_city", "destination_city",
            "distance_km", "carrier", "vehicle_type", "priority", "weight_kg", "freight_cost_inr", "delivery_status"]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest]
    num = df.select_dtypes("float").columns
    df[num] = df[num].round(4)
    for c in ["order_date", "dispatch_date", "delivery_date"]:
        df[c] = df[c].dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False)

    Path(report_dir).mkdir(exist_ok=True)
    with open(Path(report_dir) / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    make_figures(raw_numeric, df, report["raw"]["missing_by_column"],
                 report["clean"]["missing_by_column"], Path(fig_dir))

    print(json.dumps(report, indent=2))
    print(f"\nCleaned data saved to {output_path}  ({df.shape[0]} rows x {df.shape[1]} columns)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Logistics data preprocessing pipeline")
    ap.add_argument("--input", default="logistics_raw.csv")
    ap.add_argument("--output", default="logistics_cleaned.csv")
    args = ap.parse_args()
    run(args.input, args.output)
