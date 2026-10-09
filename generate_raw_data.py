"""
generate_raw_data.py
--------------------
Creates a *simulated* raw logistics dataset (logistics_raw.csv) that mimics the
structure and the typical data-quality problems of public logistics datasets
(e.g. the DataCo Smart Supply Chain dataset on Kaggle, or Indian freight /
shipment records).

Why simulate?  Real operational logistics data is rarely public and, when it is,
the quality problems are undocumented.  Here we inject known issues (missing
values, duplicates, inconsistent labels, impossible values, extreme outliers)
into otherwise realistic shipments, so the cleaning pipeline can be validated
against a known ground truth.

Run:  python generate_raw_data.py
"""
import numpy as np
import pandas as pd

SEED = 42
N = 2000                                   # unique shipments
EXTRACT_DATE = pd.Timestamp("2025-01-02")  # day the data was "exported"
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- reference
CITIES = {  # (lat, lon)
    "Delhi": (28.61, 77.21), "Mumbai": (19.08, 72.88), "Kolkata": (22.57, 88.36),
    "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.38, 78.48),
    "Patna": (25.59, 85.14), "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57),
    "Jaipur": (26.91, 75.79), "Lucknow": (26.85, 80.95),
}
HUBS = {"Delhi": .20, "Mumbai": .20, "Bengaluru": .15, "Kolkata": .12,
        "Chennai": .12, "Hyderabad": .11, "Ahmedabad": .10}
CITY_VARIANTS = {
    "Delhi": ["delhi", "DELHI", "New Delhi", "Delhi "],
    "Mumbai": ["mumbai", "Bombay", "Mumbai "],
    "Bengaluru": ["Bangalore", "bangalore", "Bengaluru "],
    "Kolkata": ["kolkata", "Calcutta"],
    "Chennai": ["chennai", "Madras"],
    "Hyderabad": ["hyderabad", "Hyderabad "],
    "Patna": ["patna", "PATNA"], "Pune": ["pune"], "Ahmedabad": ["ahmedabad"],
    "Jaipur": ["jaipur"], "Lucknow": ["lucknow", "Lucknow "],
}
CARRIERS = ["RapidFreight", "BharatCargo", "SafeHaul", "TransIndia Logistics", "GreenLine Carriers"]
CARRIER_P = [.28, .24, .20, .16, .12]
VEHICLES = ["Mini Truck", "Truck", "Container", "Air Cargo"]
SPEED = {"Mini Truck": 350, "Truck": 400, "Container": 450, "Air Cargo": 1500}   # km/day
RATE_KM = {"Mini Truck": 12, "Truck": 22, "Container": 35, "Air Cargo": 30}       # INR/km
RATE_KG = {"Mini Truck": 1.2, "Truck": 0.6, "Container": 0.4, "Air Cargo": 18}    # INR/kg
WEIGHT = {  # median kg, sigma, clip range
    "Mini Truck": (700, .6, 50, 2500), "Truck": (6000, .6, 800, 18000),
    "Container": (14000, .35, 5000, 28000), "Air Cargo": (150, 1.0, 5, 2000)}


def haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (*CITIES[a], *CITIES[b]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h)) * 1.28      # 1.28 ~ road/air-line factor


# ------------------------------------------------------------ clean "truth"
origin = rng.choice(list(HUBS), N, p=list(HUBS.values()))
dest = np.array([rng.choice([c for c in CITIES if c != o]) for o in origin])
dist = np.array([haversine(o, d) * rng.normal(1, .04) for o, d in zip(origin, dest)]).round(1)

veh = []
for d in dist:
    p = [.60, .35, .05, 0] if d < 500 else [.25, .50, .20, .05] if d < 1500 else [.05, .40, .35, .20]
    veh.append(rng.choice(VEHICLES, p=p))
veh = np.array(veh)

wt = np.array([np.clip(rng.lognormal(np.log(WEIGHT[v][0]), WEIGHT[v][1]), WEIGHT[v][2], WEIGHT[v][3])
               for v in veh]).round(1)
priority = rng.choice(["Standard", "Express"], N, p=[.75, .25])

month_w = np.array([1, 1, 1, 1, 1, 1, 1, 1.1, 1.1, 1.4, 1.5, 1.3])       # festive-season peak
days = pd.date_range("2024-01-01", "2024-12-31")
w = np.array([month_w[d.month - 1] for d in days]); w /= w.sum()
order = pd.to_datetime(rng.choice(days, N, p=w))
lag = np.where(priority == "Express", rng.choice([0, 1], N), rng.choice([0, 1, 2, 3], N, p=[.3, .4, .2, .1]))
dispatch = order + pd.to_timedelta(lag, unit="D")

expected = np.maximum(1, np.ceil(dist / np.array([SPEED[v] for v in veh])))
monsoon = np.isin(dispatch.month, [7, 8, 9])
lam = np.where(monsoon, 0.40, 0.12) * (1 + dist / 2000) * np.where(veh == "Air Cargo", .3, 1) \
      * np.where(priority == "Express", .6, 1)
transit = expected + rng.poisson(lam)
delivery = dispatch + pd.to_timedelta(transit, unit="D")

cost = (dist * np.array([RATE_KM[v] for v in veh]) + wt * np.array([RATE_KG[v] for v in veh]))
cost = (cost * np.where(priority == "Express", 1.25, 1) * rng.lognormal(0, .07, N)).round(0)

status = np.where(delivery > EXTRACT_DATE, "In Transit",
                  np.where(rng.random(N) < .025, "Returned", "Delivered"))

df = pd.DataFrame({
    "shipment_id": [f"SHP-{i:06d}" for i in range(1, N + 1)],
    "order_date": order, "dispatch_date": dispatch, "delivery_date": delivery,
    "origin_city": origin, "destination_city": dest, "distance_km": dist,
    "carrier": rng.choice(CARRIERS, N, p=CARRIER_P), "vehicle_type": veh,
    "priority": priority, "weight_kg": wt, "freight_cost_inr": cost, "delivery_status": status,
})
df.loc[df.delivery_status == "In Transit", "delivery_date"] = pd.NaT   # legitimately unknown

# ------------------------------------------------- inject data-quality issues
def pick(k, mask=None):
    pool = df.index[mask] if mask is not None else df.index
    return rng.choice(pool, k, replace=False)

# 1) impossible / extreme values (entered wrongly at source)
df.loc[pick(14), "weight_kg"] *= rng.choice([10, 100], 14)          # 250 kg typed as 25,000 kg
df.loc[pick(6), "distance_km"] = rng.choice([-1, 0, -250], 6)       # negative / zero distance
df.loc[pick(6), "distance_km"] *= 10                                  # extra zero typed
df.loc[pick(12), "freight_cost_inr"] *= rng.uniform(8, 15, 12)       # cost 10x too high
df.loc[pick(4), "freight_cost_inr"] = 0                               # zero freight cost
dl = df.delivery_status != "In Transit"
bad = pick(8, dl); df.loc[bad, "delivery_date"] = df.loc[bad, "dispatch_date"] - pd.to_timedelta(rng.integers(1, 6, 8), unit="D")
bad = pick(6, dl); df.loc[bad, "delivery_date"] = df.loc[bad, "delivery_date"] + pd.DateOffset(years=28)  # 2052 typo

# 2) convert to messy strings
def fmt_date(s):
    out = []
    for v in s:
        if pd.isna(v): out.append(np.nan); continue
        f = rng.choice(["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"], p=[.80, .12, .08])
        out.append(v.strftime(f))
    return out

raw = df.copy()
for c in ["order_date", "dispatch_date", "delivery_date"]:
    raw[c] = fmt_date(df[c])
raw["weight_kg"] = [f"{x} kg" if rng.random() < .05 else f"{x}" for x in df.weight_kg]
def fmt_cost(x):
    r = rng.random()
    return f"₹{x:,.0f}" if r < .20 else f"INR {x:,.0f}" if r < .30 else f"{x:.0f}"
raw["freight_cost_inr"] = [fmt_cost(x) for x in df.freight_cost_inr]
raw["distance_km"] = df.distance_km.astype(str)

# 3) inconsistent labels
def vary_city(c):
    return rng.choice(CITY_VARIANTS[c]) if rng.random() < .15 else c
raw["origin_city"] = [vary_city(c) for c in df.origin_city]
raw["destination_city"] = [vary_city(c) for c in df.destination_city]
raw["carrier"] = [rng.choice([c.lower(), c.upper(), f" {c}", c + " "]) if rng.random() < .08 else c for c in df.carrier]
raw["delivery_status"] = [{"Delivered": rng.choice(["Delivered", "delivered", "DELIVERED", "Delivered "], p=[.9, .04, .03, .03]),
                           "In Transit": rng.choice(["In Transit", "in transit", "In-Transit"], p=[.8, .1, .1]),
                           "Returned": rng.choice(["Returned", "returned"], p=[.9, .1])}[s] for s in df.delivery_status]

# 4) missing values (random placeholders, as seen in real exports)
TOKENS = ["", "NA", "N/A", "null", "-"]
def make_missing(col, frac, mask=None):
    idx = pick(int(frac * N), mask)
    raw.loc[idx, col] = rng.choice(TOKENS, len(idx))
make_missing("weight_kg", .06); make_missing("distance_km", .04)
make_missing("freight_cost_inr", .05); make_missing("carrier", .03)
make_missing("vehicle_type", .02); make_missing("delivery_date", .04, df.delivery_status != "In Transit")
raw.loc[raw.delivery_date.isna() & (df.delivery_status == "In Transit"), "delivery_date"] = ""  # blank = not yet delivered

# 5) duplicate records (system re-sync) -> some with different text casing
dups = raw.sample(45, random_state=SEED).copy()
dups.loc[dups.sample(15, random_state=1).index, "carrier"] = dups["carrier"].str.upper()
raw = pd.concat([raw, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)

raw.to_csv("logistics_raw.csv", index=False, encoding="utf-8")
print(f"logistics_raw.csv written: {raw.shape[0]} rows x {raw.shape[1]} columns")
