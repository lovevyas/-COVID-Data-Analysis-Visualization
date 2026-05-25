import numpy as np
from pathlib import Path

"""
numpy_ops.py  —  Task 1: NumPy Operations
------------------------------------------
Pure NumPy only — no Pandas imported here, intentionally.
Pandas is introduced properly in Task 2 (data_loader.py).

Data is loaded directly from CSV using np.genfromtxt(),
which is NumPy's own CSV reader.

Covers every topic from the assignment spec:
  ✓ Single-dimensional array
  ✓ Multi-dimensional array (ndarray)
  ✓ Mathematical operations
  ✓ Vectorized operations
  ✓ Broadcasting
  ✓ Slicing & indexing
  ✓ Reshaping arrays
  ✓ Splitting arrays

Why np.genfromtxt over pd.read_csv here:
  Task 1 is about learning NumPy fundamentals.
  Pandas (pd.read_csv) is introduced in Task 2 where it belongs.
  This keeps the learning progression honest and consistent.
"""

import numpy as np
from pathlib import Path

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
print(BASE_DIR)
DATA_PATH = BASE_DIR / "data" / "country_wise_latest.csv"


# ── CSV loading (pure NumPy) ──────────────────────────────────────────────────

def _load_numeric() -> np.ndarray:
    """
    Load 4 numeric columns directly into a NumPy array.

    Columns extracted (by index from CSV):
      1 → Confirmed
      2 → Deaths
      3 → Recovered
      4 → Active

    We deliberately skip columns with 'inf' values
    (e.g. col 10: Deaths/100 Recovered) — those are
    handled properly in Task 3 (data cleaning).

    Returns
    -------
    np.ndarray, shape (187, 4), dtype float64
    """
    return np.genfromtxt(
        DATA_PATH,
        delimiter=",",
        skip_header=1,          # skip the column-name row
        usecols=(1, 2, 3, 4),   # Confirmed, Deaths, Recovered, Active
        filling_values=np.nan,  # any unreadable cell becomes nan
        dtype=np.float64,
    )


def _load_countries() -> np.ndarray:
    """
    Load country names as a 1-D string array.
    Kept separate because genfromtxt can't mix str + float in one call
    without a structured dtype — keeping them separate is cleaner.

    Returns
    -------
    np.ndarray, shape (187,), dtype '<U...' (unicode string)
    """
    return np.genfromtxt(
        DATA_PATH,
        delimiter=",",
        skip_header=1,
        usecols=(0,),
        dtype=str,
    )


# ── Section header helper ─────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ── Topic 1: Single-dimensional array ────────────────────────────────────────

def demo_1d_array(matrix: np.ndarray) -> np.ndarray:
    """
    Extract 'Confirmed' cases as a 1-D array from column 0 of our matrix.

    This is the entry point — a single column slice of the 2-D matrix.
    Every subsequent demo either uses this directly or builds on it.
    """
    _section("1. Single-Dimensional Array — Confirmed Cases")

    confirmed = matrix[:, 0]    # all rows, first column

    print(f"Type            : {type(confirmed)}")
    print(f"Shape           : {confirmed.shape}")       # (187,)
    print(f"Dimensions      : {confirmed.ndim}")        # 1
    print(f"Dtype           : {confirmed.dtype}")
    print(f"First 5 values  : {confirmed[:5]}")
    print(f"Total countries : {len(confirmed)}")

    return confirmed


# ── Topic 2: Multi-dimensional array ─────────────────────────────────────────

def demo_2d_array(matrix: np.ndarray) -> None:
    """
    Show the full (187, 4) matrix — this IS the ndarray.

    Columns: [Confirmed | Deaths | Recovered | Active]
    Rows   : one per country (187 total)

    Why 2-D matters: this is the shape every ML library expects.
    Rows = observations, columns = features.
    """
    _section("2. Multi-Dimensional Array (ndarray) — 4 Key Metrics")

    print(f"Shape           : {matrix.shape}")          # (187, 4)
    print(f"Dimensions      : {matrix.ndim}")           # 2
    print(f"Dtype           : {matrix.dtype}")
    print(f"Total elements  : {matrix.size}")           # 748
    print(f"Memory (bytes)  : {matrix.nbytes}")

    print(f"\nColumn layout   : [Confirmed | Deaths | Recovered | Active]")
    print(f"First 3 rows    :")
    print(matrix[:3])


# ── Topic 3: Mathematical operations ─────────────────────────────────────────

def demo_math_operations(confirmed: np.ndarray) -> None:
    """
    Core aggregation functions on the Confirmed column.

    np.nan* variants used throughout because some entries
    could be nan (filling_values=np.nan in genfromtxt).
    Plain np.mean() returns nan for the whole array if even
    one nan exists — a very common real-world bug.
    """
    _section("3. Mathematical Operations — Confirmed Cases")

    print(f"Sum              : {np.nansum(confirmed):>15,.0f}")
    print(f"Mean             : {np.nanmean(confirmed):>15,.2f}")
    print(f"Median           : {np.nanmedian(confirmed):>15,.2f}")
    print(f"Max              : {np.nanmax(confirmed):>15,.0f}")
    print(f"Min              : {np.nanmin(confirmed):>15,.0f}")
    print(f"Std deviation    : {np.nanstd(confirmed):>15,.2f}")
    print(f"Variance         : {np.nanvar(confirmed):>15,.2f}")

    p25, p50, p75 = np.nanpercentile(confirmed, [25, 50, 75])
    print(f"\nPercentiles:")
    print(f"  25th  : {p25:>12,.0f}")
    print(f"  50th  : {p50:>12,.0f}  ← same as median")
    print(f"  75th  : {p75:>12,.0f}")
    print(f"\nIQR (75th - 25th) : {p75 - p25:>10,.0f}")

    # np.cumsum — running total across countries
    cumulative = np.nancumsum(confirmed)
    print(f"\nCumulative sum (last 3 values): {cumulative[-3:]}")
    print(f"→ Final value matches total sum: {cumulative[-1]:,.0f}")


# ── Topic 4: Vectorized operations ───────────────────────────────────────────

def demo_vectorized_operations(matrix: np.ndarray) -> np.ndarray:
    """
    Compute Case Fatality Rate (CFR) for all 187 countries in ONE line.
    CFR = (Deaths / Confirmed) × 100

    Vectorized means: no Python for-loop, operation applied to every
    element simultaneously using compiled C code under the hood.
    np.where handles the edge case where Confirmed == 0 safely.
    """
    _section("4. Vectorized Operations — Case Fatality Rate per Country")

    confirmed = matrix[:, 0]
    deaths    = matrix[:, 1]

    # One expression, 187 results — this is vectorization
    cfr = np.where(confirmed > 0, (deaths / confirmed) * 100, 0.0)

    print("CFR = (Deaths / Confirmed) × 100  — computed for all 187 at once")
    print(f"\nFirst 5 CFR values : {cfr[:5].round(2)}")
    print(f"Global average CFR : {np.nanmean(cfr):.2f}%")
    print(f"Highest CFR        : {np.nanmax(cfr):.2f}%")
    print(f"Lowest CFR (> 0)   : {cfr[cfr > 0].min():.4f}%")

    # Additional vectorized op: recovery rate
    recovered = matrix[:, 2]
    recovery_rate = np.where(confirmed > 0, (recovered / confirmed) * 100, 0.0)
    print(f"\nRecovery rate (vectorized):")
    print(f"  Global average   : {np.nanmean(recovery_rate):.2f}%")
    print(f"  Highest          : {np.nanmax(recovery_rate):.2f}%")

    return cfr


# ── Topic 5: Broadcasting ─────────────────────────────────────────────────────

def demo_broadcasting(matrix: np.ndarray) -> np.ndarray:
    """
    Broadcasting: operate on arrays of different shapes without copying data.

    matrix shape  : (187, 4)
    col_max shape :      (4,)   ← one max per column

    NumPy 'stretches' (4,) across all 187 rows automatically.
    No loop, no manual tiling — this is what broadcasting IS.

    This is exactly how min-max feature scaling works in scikit-learn.
    """
    _section("5. Broadcasting — Min-Max Normalisation across all Metrics")

    col_min = np.nanmin(matrix, axis=0)    # shape (4,) — per column min
    col_max = np.nanmax(matrix, axis=0)    # shape (4,) — per column max

    print(f"matrix shape    : {matrix.shape}")
    print(f"col_max shape   : {col_max.shape}  ← broadcasts across 187 rows")
    print(f"\nColumn maxima:")
    labels = ["Confirmed", "Deaths", "Recovered", "Active"]
    for label, mn, mx in zip(labels, col_min, col_max):
        print(f"  {label:<12}: min={mn:>10,.0f}  max={mx:>12,.0f}")

    # (187,4) - (4,) and / (4,) — both broadcast automatically
    normalised = (matrix - col_min) / (col_max - col_min)

    print(f"\nNormalised shape: {normalised.shape}  (unchanged)")
    print(f"All values in [0, 1]: {normalised.min():.4f} → {normalised.max():.4f}")
    print(f"\nFirst 3 rows normalised [Confirmed | Deaths | Recovered | Active]:")
    print(normalised[:3].round(4))

    return normalised


# ── Topic 6: Slicing & Indexing ───────────────────────────────────────────────

def demo_slicing_indexing(confirmed: np.ndarray, countries: np.ndarray) -> None:
    """
    All major slicing and indexing patterns:
      - Basic slice        arr[start:stop:step]
      - Negative index     arr[-n]
      - Boolean mask       arr[condition]
      - Fancy indexing     arr[[i, j, k]]
      - np.argmax/argmin   find index of extreme values
      - np.argsort         sort-order indices
    """
    _section("6. Slicing & Indexing")

    # Basic slice
    print("── Basic slice ──────────────────────────────────────")
    print(f"First 5     : {confirmed[:5]}")
    print(f"Last 5      : {confirmed[-5:]}")
    print(f"Every 30th  : {confirmed[::30]}")
    print(f"Reversed    : {confirmed[::-1][:5]}  (first 5 of reversed)")

    # Negative indexing
    print(f"\n── Negative indexing ────────────────────────────────")
    print(f"Last country      : {countries[-1]}  →  {confirmed[-1]:,.0f}")
    print(f"Second to last    : {countries[-2]}  →  {confirmed[-2]:,.0f}")

    # Boolean mask — high burden countries
    print(f"\n── Boolean mask: Confirmed > 100,000 ────────────────")
    mask         = confirmed > 100_000
    high_c       = countries[mask]
    high_v       = confirmed[mask]
    print(f"Count  : {mask.sum()} countries")
    for c, v in zip(high_c[:6], high_v[:6]):
        print(f"  {c:<35} {v:>10,.0f}")

    # Fancy indexing
    print(f"\n── Fancy indexing: hand-picked positions ────────────")
    idx = [0, 25, 50, 100, 150, 186]
    for i in idx:
        print(f"  [{i:>3}]  {countries[i]:<35} {confirmed[i]:>10,.0f}")

    # np.argmax / np.argmin
    print(f"\n── np.argmax / np.argmin ────────────────────────────")
    hi_idx = np.argmax(confirmed)
    lo_idx = np.argmin(confirmed)
    print(f"Most  confirmed : {countries[hi_idx]:<30} {confirmed[hi_idx]:>10,.0f}")
    print(f"Least confirmed : {countries[lo_idx]:<30} {confirmed[lo_idx]:>10,.0f}")

    # np.argsort — top 5 descending
    print(f"\n── np.argsort: Top 5 most confirmed ────────────────")
    top5 = np.argsort(confirmed)[::-1][:5]
    for rank, i in enumerate(top5, 1):
        print(f"  #{rank}  {countries[i]:<35} {confirmed[i]:>12,.0f}")


# ── Topic 7: Reshaping ────────────────────────────────────────────────────────

def demo_reshaping(matrix: np.ndarray) -> None:
    """
    reshape() changes the view of the data — no data is copied.
    The only rule: total element count must stay the same.

    Practical use:
      - Flattening before feeding into ML models
      - Adding dimensions with np.newaxis for broadcasting
      - Grouping rows into batches (common in deep learning)
    """
    _section("7. Reshaping Arrays")

    print(f"Original shape  : {matrix.shape}  → 187 countries × 4 metrics")

    # Flatten — (748,)
    flat = matrix.flatten()
    print(f"flatten()       : {flat.shape}  → {flat.size} values in sequence")

    # Ravel — same as flatten but returns a view, not a copy (faster)
    ravelled = matrix.ravel()
    print(f"ravel()         : {ravelled.shape}  → view (no copy, more efficient)")

    # 3-D reshape: 11 × 17 = 187 exactly
    reshaped_3d = matrix.reshape(11, 17, 4)
    print(f"reshape(11,17,4): {reshaped_3d.shape}  → 11 groups of 17 countries")
    print(f"  Group 0, Country 0, all metrics : {reshaped_3d[0, 0, :]}")
    print(f"  Group 1, Country 0, all metrics : {reshaped_3d[1, 0, :]}")

    # Transpose — flip axes: (187,4) → (4,187)
    transposed = matrix.T
    print(f"transpose (.T)  : {transposed.shape}  → 4 metrics × 187 countries")
    print(f"  Row 0 (Confirmed), first 5: {transposed[0, :5]}")

    # np.newaxis — insert a new axis (very common in ML)
    expanded = matrix[:5, np.newaxis, :]
    print(f"np.newaxis      : {matrix[:5].shape} → {expanded.shape}")
    print(f"  Used to align shapes for broadcasting in higher-dim operations")


# ── Topic 8: Splitting ────────────────────────────────────────────────────────

def demo_splitting(matrix: np.ndarray, countries: np.ndarray) -> None:
    """
    np.array_split — split into N parts along an axis.

    Unlike np.split(), array_split handles uneven sizes gracefully.
    np.split() raises an error if 187 doesn't divide evenly — it doesn't.

    np.hsplit — split column-wise (horizontally).
    Gives each metric its own standalone array.

    Practical use: train/val/test splits, batching, parallel processing.
    """
    _section("8. Splitting Arrays")

    # Row-wise split into 3 parts (simulating train / val / test)
    print("── np.array_split: row-wise into 3 parts ────────────")
    parts  = np.array_split(matrix, 3, axis=0)
    clabels = ["Train (part A)", "Val   (part B)", "Test  (part C)"]
    for label, part in zip(clabels, parts):
        print(f"  {label}: {part.shape}  "
              f"| Confirmed sum = {part[:, 0].sum():>12,.0f}")

    # Column-wise split — each metric becomes its own array
    print(f"\n── np.hsplit: column-wise → 4 separate metric arrays ──")
    confirmed_arr, deaths_arr, recovered_arr, active_arr = np.hsplit(matrix, 4)
    col_names = ["Confirmed", "Deaths", "Recovered", "Active"]
    for name, arr in zip(col_names, [confirmed_arr, deaths_arr, recovered_arr, active_arr]):
        print(f"  {name:<12}: shape={arr.shape}  "
              f"sum={arr.sum():>12,.0f}  mean={arr.mean():>8,.1f}")

    # np.vsplit — vertical split (same as array_split on axis=0 for 2-D)
    print(f"\n── np.vsplit: first 170 vs last 17 countries ────────")
    top, bottom = np.array_split(matrix, [170], axis=0)
    print(f"  Top block    : {top.shape}")
    print(f"  Bottom block : {bottom.shape}")
    print(f"  Bottom countries sample: {countries[-3:].tolist()}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all() -> dict:
    """
    Run all 8 NumPy demos in sequence.
    Returns computed arrays for potential reuse by later tasks.

    Note: this function loads data itself using pure NumPy.
    From Task 2 onward, data_loader.py (Pandas) takes over.
    """
    print("\n" + "█" * 60)
    print("  TASK 1 — NumPy Operations on COVID-19 Data")
    print("  (Pure NumPy — no Pandas)")
    print("█" * 60)

    # Load once — pass arrays into each demo
    matrix    = _load_numeric()      # (187, 4) — Confirmed, Deaths, Recovered, Active
    countries = _load_countries()    # (187,)   — country name strings
    confirmed = matrix[:, 0]        # 1-D view, not a copy

    # Run all topics
    demo_1d_array(matrix)
    demo_2d_array(matrix)
    demo_math_operations(confirmed)
    cfr        = demo_vectorized_operations(matrix)
    normalised = demo_broadcasting(matrix)
    demo_slicing_indexing(confirmed, countries)
    demo_reshaping(matrix)
    demo_splitting(matrix, countries)

    print("\n" + "█" * 60)
    print("  Task 1 complete.")
    print("  Next: Task 2 introduces Pandas via data_loader.py")
    print("█" * 60 + "\n")

    return {
        "matrix"    : matrix,
        "countries" : countries,
        "confirmed" : confirmed,
        "cfr"       : cfr,
        "normalised": normalised,
    }


if __name__ == "__main__":
    run_all()