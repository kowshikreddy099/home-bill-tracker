"""Creates bills.csv: 30 months of example household bills (replace with your real data)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
months = pd.period_range(end=pd.Timestamp.now().to_period("M"), periods=30, freq="M")   # 30 months ending this month

rows = []
for i, m in enumerate(months):
    season = np.cos((m.month - 5) / 12 * 2 * np.pi)          # peaks in May (summer, AC use)
    elec_units = 190 + 60 * season + rng.normal(0, 12) + i * 0.8
    water_units = 13 + 2 * season + rng.normal(0, 0.8)
    rows.append({
        "month": str(m),
        "rent": 18000 if i < 12 else 19000,                   # rent revision after a year
        "electricity": 50 + max(0, min(elec_units, 100)) * 3.5
                          + max(0, min(elec_units, 200) - 100) * 5
                          + max(0, elec_units - 200) * 7.5,
        "water": 80 + water_units * 12,
        "internet": 799 if i < 8 else 899,
        "streaming": 499 if i < 14 else 649,
    })

pd.DataFrame(rows).round(2).to_csv("bills.csv", index=False)
print("Wrote bills.csv with", len(rows), "months")
