# 🚚 Logistics Data Cleaning & Preprocessing Pipeline

A reproducible Python pipeline that turns a messy, shipment-level logistics export into an analysis-ready dataset.
Built for **Week 2: Data Collection, Cleaning and Preprocessing for Logistics Analysis**.

> **Note on the data.** Real operational logistics data is rarely public, so `logistics_raw.csv` is a **simulated**
> dataset (2,045 rows) modelled on the structure of public supply-chain datasets such as the
> [DataCo Smart Supply Chain dataset](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
> and Indian freight records. Realistic data-quality problems were *deliberately injected*, which lets the pipeline be
> validated against a known ground truth. `generate_raw_data.py` recreates it (seed = 42).

## What the pipeline does

| # | Step | Techniques |
|---|------|-----------|
| 1 | Load & audit | read as text, placeholder tokens (`NA`, `N/A`, `null`, `-`, empty) → `NaN` |
| 2 | Standardise | city aliases (Bombay→Mumbai), case/whitespace, `₹12,450`→`12450`, 3 date formats parsed explicitly |
| 3 | De-duplicate | one row per `shipment_id`, keeping the most complete record |
| 4 | Business-rule validation | distance ≤ 0 or > 4,000 km, weight > vehicle capacity, cost ≤ 0, delivery before dispatch / > 30 days |
| 5 | Missing-value imputation | route median (distance), vehicle-type median (weight), cost-per-km model (cost), expected transit (delivery date); `Unknown` for categoricals; imputation flags kept |
| 6 | Outlier handling | weight: IQR on log scale per vehicle (cap) · cost: regression residual + MAD (correct) · delays: flag only |
| 7 | Feature engineering | `transit_days`, `expected_transit_days`, `delay_days`, `is_late`, `cost_per_km`, `dispatch_lag_days` |
| 8 | Normalisation | `log1p` + Min-Max for skewed values, Z-score for transit / delay |

## Results (raw → cleaned)

| Metric | Raw | Cleaned |
|---|---|---|
| Rows | 2,045 | 2,000 |
| Duplicate shipment IDs | 45 | 0 |
| Distinct spellings – destination city | 34 | 11 |
| Distinct spellings – carrier | 26 | 5 (+ `Unknown`) |
| Distinct spellings – delivery status | 9 | 3 |
| Impossible values | 40 | 0 |
| Max weight (kg) | 2,198,860 | 28,000 |
| Max freight cost (INR) | 1,468,782 | 122,972 |
| Missing values in key fields | yes | 0 (21 `In Transit` rows correctly keep an empty delivery date) |

Dropping every incomplete row instead of imputing would have discarded **23.6 %** of the shipments.

## Quick start

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt

python generate_raw_data.py            # (optional) re-create logistics_raw.csv
python logistics_preprocessing.py      # -> logistics_cleaned.csv, reports/, figures/
```

Custom paths: `python logistics_preprocessing.py --input my_raw.csv --output my_clean.csv`

## Repository structure

```
├── README.md
├── requirements.txt
├── generate_raw_data.py            # builds the simulated raw dataset
├── logistics_preprocessing.py      # the full pipeline (one function per step)
├── logistics_raw.csv               # input  – messy data
├── logistics_cleaned.csv           # output – analysis-ready data (32 columns)
├── Week2_Logistics_Project.docx    # full written report
├── reports/data_quality_report.json
└── figures/                        # before/after charts used in the report
```

## Columns added in `logistics_cleaned.csv`

| Group | Columns |
|---|---|
| Audit flags | `distance_imputed`, `weight_imputed`, `cost_imputed`, `delivery_imputed`, `weight_kg_outlier`, `cost_outlier`, `severe_delay_flag` |
| Features | `dispatch_lag_days`, `transit_days`, `expected_transit_days`, `delay_days`, `is_late`, `cost_per_km`, `order_month` |
| Scaled | `weight_kg_norm`, `distance_km_norm`, `freight_cost_inr_norm` (0–1) · `transit_days_z`, `delay_days_z` (mean 0, sd 1) |

Original-scale columns are kept next to the scaled ones, so results stay interpretable.

## Design decisions worth knowing

* **Missing ≠ wrong.** A shipment that is still `In Transit` has no delivery date *by definition*; it is never imputed.
* **Outliers are not all errors.** A 28-tonne container is real; a 250 kg parcel typed as 25,000 kg is not. Domain rules
  (vehicle capacity) come first, statistics second, and genuine delays are *flagged but kept*, because they are the signal
  logistics managers care about.
* **No silent changes.** Every imputed or corrected value is flagged, so downstream models can include or exclude them.

## Limitations

* The data is simulated; real data will have messier, less predictable errors.
* Median imputation shrinks variance slightly; for modelling, consider multiple imputation or model-based imputation.
* Fit scalers on the *training split only* when you use this for machine learning, to avoid data leakage.

## Tech stack

Python 3 · pandas · NumPy · scikit-learn · Matplotlib
