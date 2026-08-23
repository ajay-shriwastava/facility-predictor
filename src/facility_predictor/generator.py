"""
Synthetic booking data generator.

Produces realistic 2025 facility-booking history for 200 residents
across 7 facilities in a residential community.

Output schema: booking_id, resident_id, facility_id, booking_timestamp, usage_timestamp
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
RNG = np.random.default_rng(SEED)
Faker.seed(SEED)
_fake = Faker()

# ── Constants ──────────────────────────────────────────────────────────────────

YEAR = 2025
START_DATE = datetime(YEAR, 1, 2)        # Jan 2 — leaves room for lead times
END_DATE = datetime(YEAR, 12, 31, 20, 0)

FACILITIES = [
    "Gym",
    "Swimming Pool",
    "Badminton Court",
    "Tennis Court",
    "Clubhouse",
    "Multipurpose Hall",
    "Kids Play Area",
]
FAC_WEIGHTS = np.array([0.30, 0.22, 0.18, 0.12, 0.10, 0.05, 0.03])

# Max concurrent bookings allowed per facility per hour slot (hard capacity limit)
CAPACITY: dict[str, int] = {
    "Gym": 25,              # large gym floor, multiple equipment stations
    "Swimming Pool": 20,    # lane-based, safety cap
    "Badminton Court": 4,   # 2 courts × 2 players (singles) or doubles pairs
    "Tennis Court": 4,      # 2 courts × 2 players
    "Clubhouse": 40,        # lounge/event space
    "Multipurpose Hall": 80,  # largest venue
    "Kids Play Area": 15,   # supervised area, safety limit
}

NOISE_PROB = 0.05
POOL_MONTHS = {4, 5, 6, 7, 8, 9}  # Apr–Sep: swimming season

# ── Archetype specifications ───────────────────────────────────────────────────

_ARCHETYPES = [
    dict(
        name="morning_gym", count=50,
        facilities=["Gym"], fac_w=[1.0],
        hours=(6, 9), days=[0, 1, 2, 3, 4],
        bookings=(4, 5), lead_mu=20, lead_sigma=3,
        skip_wk=0.05, skip_mo=0.05,
    ),
    dict(
        name="weekend_social", count=40,
        facilities=["Swimming Pool", "Clubhouse"], fac_w=[0.6, 0.4],
        hours=(14, 18), days=[5, 6],
        bookings=(1, 2), lead_mu=24, lead_sigma=8,
        skip_wk=0.15, skip_mo=0.10,
    ),
    dict(
        name="evening_sports", count=40,
        facilities=["Badminton Court", "Tennis Court"], fac_w=[0.65, 0.35],
        hours=(18, 21), days=[0, 1, 2, 3, 4, 5],
        bookings=(2, 3), lead_mu=9, lead_sigma=2,
        skip_wk=0.10, skip_mo=0.08,
    ),
    dict(
        name="occasional", count=35,
        facilities=FACILITIES, fac_w=FAC_WEIGHTS.tolist(),
        hours=(8, 20), days=list(range(7)),
        bookings=(0, 1), lead_mu=4, lead_sigma=2,
        skip_wk=0.50, skip_mo=0.15,
    ),
    dict(
        name="seasonal_swimmer", count=20,
        facilities=["Swimming Pool", "Gym"], fac_w=None,
        hours=(17, 20), days=[1, 3, 5],
        bookings=(2, 3), lead_mu=18, lead_sigma=4,
        skip_wk=0.10, skip_mo=0.08,
    ),
    dict(
        name="drifter", count=15,
        facilities=None, fac_w=None,
        hours=(9, 21), days=list(range(7)),
        bookings=(1, 2), lead_mu=24, lead_sigma=10,
        skip_wk=0.20, skip_mo=0.10,
    ),
]


# ── Resident builder ───────────────────────────────────────────────────────────

def _build_residents() -> list[dict]:
    """Assign each of the 200 residents an archetype and personal parameters."""
    residents = []
    rid = 1

    for spec in _ARCHETYPES:
        for _ in range(spec["count"]):
            r: dict = {"resident_id": f"R-{rid:03d}", "archetype": spec["name"], "spec": spec}

            if spec["name"] == "drifter":
                non_misc = FACILITIES[:5]
                r["initial_fac"] = str(RNG.choice(non_misc))
                remaining = [f for f in non_misc if f != r["initial_fac"]]
                r["drift_fac"] = str(RNG.choice(remaining))
                r["drift_month"] = int(RNG.integers(4, 9))

            residents.append(r)
            rid += 1

    return residents


# ── Facility sampler ───────────────────────────────────────────────────────────

def _sample_facility(resident: dict, month: int, is_noise: bool) -> str:
    if is_noise:
        return str(RNG.choice(FACILITIES, p=FAC_WEIGHTS))

    spec = resident["spec"]
    name = spec["name"]

    if name == "seasonal_swimmer":
        return "Swimming Pool" if month in POOL_MONTHS else "Gym"

    if name == "drifter":
        if month >= resident["drift_month"]:
            return resident["drift_fac"] if RNG.random() < 0.80 else resident["initial_fac"]
        return resident["initial_fac"]

    facs = spec["facilities"]
    w = np.array(spec["fac_w"])
    return str(RNG.choice(facs, p=w / w.sum()))


# ── Single booking generator ───────────────────────────────────────────────────

def _make_booking(resident: dict, week_start: datetime) -> dict | None:
    spec = resident["spec"]

    usage_day = week_start + timedelta(days=int(RNG.choice(spec["days"])))
    usage_hour = int(RNG.integers(spec["hours"][0], spec["hours"][1]))
    usage_min = int(RNG.choice([0, 15, 30, 45]))
    usage_ts = usage_day.replace(hour=usage_hour, minute=usage_min, second=0, microsecond=0)

    if usage_ts < START_DATE or usage_ts > END_DATE:
        return None

    lead_hrs = max(1.0, float(RNG.normal(spec["lead_mu"], spec["lead_sigma"])))
    booking_ts = usage_ts - timedelta(hours=lead_hrs)

    if booking_ts < datetime(YEAR, 1, 1):
        booking_ts = datetime(YEAR, 1, 1, 0, int(RNG.integers(0, 60)))

    is_noise = RNG.random() < NOISE_PROB
    facility = _sample_facility(resident, usage_ts.month, is_noise)

    return dict(
        resident_id=resident["resident_id"],
        facility_id=facility,
        booking_timestamp=booking_ts,
        usage_timestamp=usage_ts,
    )


# ── Week iterator ──────────────────────────────────────────────────────────────

def _iter_weeks() -> list[datetime]:
    """Return the Monday of every ISO week that falls within 2025."""
    weeks: list[datetime] = []
    current = START_DATE - timedelta(days=START_DATE.weekday())
    while current <= END_DATE:
        weeks.append(current)
        current += timedelta(weeks=1)
    return weeks


# ── Public API ─────────────────────────────────────────────────────────────────

def generate(output_path: str | Path | None = None) -> pd.DataFrame:
    """
    Generate the synthetic booking dataset.

    Parameters
    ----------
    output_path : path to write CSV, or None to skip writing.

    Returns
    -------
    pd.DataFrame with columns:
        booking_id, resident_id, facility_id, booking_timestamp, usage_timestamp
    """
    residents = _build_residents()
    weeks = _iter_weeks()
    rows: list[dict] = []

    for resident in residents:
        spec = resident["spec"]

        inactive_months: set[int] = set()
        for month in range(1, 13):
            if RNG.random() < spec["skip_mo"]:
                inactive_months.add(month)

        for week_start in weeks:
            if week_start.month in inactive_months:
                continue
            if RNG.random() < spec["skip_wk"]:
                continue

            n = int(RNG.integers(spec["bookings"][0], spec["bookings"][1] + 1))
            for _ in range(n):
                booking = _make_booking(resident, week_start)
                if booking:
                    rows.append(booking)

    df = pd.DataFrame(rows)

    # Sort by booking_timestamp first so FCFS determines who gets the slot
    df = df.sort_values("booking_timestamp").reset_index(drop=True)

    # Enforce capacity: within each (facility, date, hour) slot keep only the
    # first CAPACITY[facility] bookings — the rest are rejected as fully booked.
    df["_usage_date"] = df["usage_timestamp"].dt.date
    df["_usage_hour"] = df["usage_timestamp"].dt.hour
    slot_rank = df.groupby(["facility_id", "_usage_date", "_usage_hour"]).cumcount()
    slot_cap = df["facility_id"].map(CAPACITY)
    df = df[slot_rank < slot_cap].drop(columns=["_usage_date", "_usage_hour"])
    df = df.reset_index(drop=True)

    df.insert(0, "booking_id", [f"BK-{i + 1:05d}" for i in range(len(df))])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df):,} bookings to {output_path}")

    return df


if __name__ == "__main__":
    df = generate("data/synthetic_bookings.csv")
    print(f"Generated {len(df):,} bookings for {df['resident_id'].nunique()} residents")
    print(df["archetype"].value_counts() if "archetype" in df.columns else df["facility_id"].value_counts())