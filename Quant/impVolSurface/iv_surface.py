
import warnings
import numpy as np
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import date
from typing import Tuple, cast

# Local imports
from bsmFuncs import bsm_callImpVol          # your existing BSM solver
from dataExtractor import load_surface_inputs

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Core: build the raw IV DataFrame
# ---------------------------------------------------------------------------

def build_iv_dataframe(
    underlying: str = "AAPL US Equity",
    option_type: str = "C",
    rate_ticker: str = "US0003M Index",
    min_expiry_days: int = 7,
    max_expiry_days: int = 365,
    moneyness_band: float = 0.30,
) -> Tuple[pd.DataFrame, float]:
  
    
    #Pull Bloomberg data and build the initial DataFrame with spot, rate, div_yield, and option chain
    """
    load_surface_inputs(
        underlying=underlying,
        option_type=option_type,
        rate_ticker=rate_ticker,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        moneyness_band=moneyness_band,
    ) 
    """
    
    inputs = pd.read_parquet(BASE_DIR / f"data/{underlying}_{option_type}_surface_inputs.parquet").to_dict(orient="index")[0]
    
    chain: pd.DataFrame = pd.read_parquet(BASE_DIR / f"data/{underlying}_{option_type}_surface_chain.parquet")
    S        = inputs["spot"]
    r        = inputs["rate"]
    q        = inputs["div_yield"]
    today    = date.today()

    print(f"\n[surface] S={S:.2f}  r={r:.4f}  q={q:.4f}")
    print(f"[surface] Solving implied vol for {len(chain)} contracts …\n")

    records = []
    n_expired = n_no_price = n_solver_fail = n_bounds_fail = 0

    for _, row in chain.iterrows():
        dte = (pd.Timestamp(row["expiry"]).date() - today).days #type: ignore {pandas Timestamp accepts date objects but Pylance doesn't know that}
        T   = dte / 365.0

        if T <= 0:
            n_expired += 1
            continue

        px_mid = row["px_mid"]
        if px_mid is None or (isinstance(px_mid, float) and np.isnan(px_mid)) or px_mid <= 0:
            n_no_price += 1
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                iv = bsm_callImpVol(
                    S=S,
                    K=float(row["strike"]),
                    T=T,
                    r=r,
                    C=float(px_mid),
                )
        except Exception as e:
            n_solver_fail += 1
            iv = np.nan

        if iv is None or np.isnan(iv) or iv <= 0 or iv > 5.0:
            n_bounds_fail += 1
            continue

        records.append(
            {
                "expiry":         row["expiry"],
                "strike":         float(row["strike"]),
                "days_to_expiry": dte,
                "T":              T,
                "iv":             iv,
                "px_mid":         float(px_mid),
                "bbg_ivol":       row["ivol_mid"] if "ivol_mid" in row.index else np.nan,
            }
        )

    print(f"[surface] Dropped — expired: {n_expired}, no price: {n_no_price}, "
          f"solver error: {n_solver_fail}, out of bounds: {n_bounds_fail}")

    if not records:
        print("[surface] WARNING: no valid IV nodes. Sample chain rows:")
        print(chain[["expiry", "strike", "px_mid", "days_to_expiry"]].head(10).to_string())
        empty = pd.DataFrame(columns=["expiry", "strike", "days_to_expiry", "T", "iv", "px_mid", "bbg_ivol"])
        return empty, S

    iv_df = pd.DataFrame(records)
    iv_df.sort_values(["expiry", "strike"], inplace=True)
    iv_df.reset_index(drop=True, inplace=True)

    print(f"[surface] {len(iv_df)} valid IV nodes computed.")
    return iv_df, S


# ---------------------------------------------------------------------------
# Pivot to a surface matrix
# ---------------------------------------------------------------------------

def pivot_surface(iv_df: pd.DataFrame) -> pd.DataFrame:

    iv_df = iv_df.copy()
    iv_df["expiry_label"] = (
        pd.to_datetime(iv_df["expiry"]).dt.strftime("%Y-%m-%d")
        + "\n("
        + iv_df["days_to_expiry"].astype(str)
        + "d)"
    )

    surface = iv_df.pivot_table(
        index="strike",
        columns="expiry_label",
        values="iv",
        aggfunc="mean",
    )
    return surface


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_iv_surface(iv_df: pd.DataFrame, spot: float, underlying: str = "AAPL US Equity", typ: str = "C"):

    fig = plt.figure(figsize=(18, 7))
    fig.suptitle(f"BSM Implied Volatility Surface — {underlying}  (S={spot:.2f})",
                 fontsize=14, fontweight="bold")

    # ---- 3-D surface -------------------------------------------------------
    ax3d = fig.add_subplot(121, projection="3d")

    # Use T (years) for the x-axis and moneyness K/S for y
    X = iv_df["T"].to_numpy()
    Y = (iv_df["strike"] / spot).to_numpy()        # moneyness K/S
    Z = iv_df["iv"].to_numpy() * 100             # in percent

    # Scatter (reliable even with sparse data)
    sc = ax3d.scatter(X, Y, Z, c=Z, cmap='RdYlGn_r', s=18, alpha=0.85) #type: ignore {matplotlib scatter typehints int but accepts array-like}
    fig.colorbar(sc, ax=ax3d, shrink=0.5, label="IV (%)")

    ax3d.set_xlabel("Time to Expiry (yrs)", labelpad=8)
    ax3d.set_ylabel("Moneyness  K / S", labelpad=8)
    ax3d.set_zlabel("Implied Vol (%)", labelpad=8)
    ax3d.set_title("3-D Scatter")
    ax3d.view_init(elev=25, azim=-55)

    # ---- 2-D smile per expiry ----------------------------------------------
    ax2d = fig.add_subplot(122)

    expiries = sorted(iv_df["expiry"].unique())
    cmap_lines = plt.get_cmap("tab10") #type:ignore {matplotlib get_cmap exists but Pylance doesn't know that}

    for i, exp in enumerate(expiries):
        sub = iv_df[iv_df["expiry"] == exp].sort_values("strike")
        moneyness = sub["strike"] / spot
        label = f"{exp}  ({sub['days_to_expiry'].iloc[0]}d)"
        ax2d.plot(moneyness, sub["iv"] * 100,
                  marker="o", markersize=4,
                  color=cmap_lines(i % 10),
                  label=label, linewidth=1.4)

    ax2d.axvline(1.0, color="grey", linestyle="--", linewidth=0.9, label="ATM (K/S=1)")
    ax2d.set_xlabel("Moneyness  K / S")
    ax2d.set_ylabel("Implied Volatility (%)")
    ax2d.set_title("Smile per Expiry")
    ax2d.legend(fontsize=7, loc="upper right", ncol=2)
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(BASE_DIR / f"outputs/{underlying}_{typ}_iv_surface.png", dpi=150, bbox_inches="tight")
    print(f"\n[surface] Plot saved → {underlying}_{typ}_iv_surface.png")
    plt.show()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def surface_summary(iv_df: pd.DataFrame, underlying: str):
    """Print a concise summary table of average IV per expiry."""
    summary = (
        iv_df.groupby(["expiry", "days_to_expiry"])["iv"]
        .agg(["mean", "min", "max", "count"])
        .rename(columns={"mean": "avg_iv", "min": "min_iv", "max": "max_iv", "count": "n_strikes"})
        .reset_index()
    )
    summary[["avg_iv", "min_iv", "max_iv"]] *= 100   # convert to %
    summary.columns = ["Expiry", "DTE", "Avg IV %", "Min IV %", "Max IV %", "# Strikes"]
    print("\n" + "=" * 62)
    print(f"  IV Surface Summary — {underlying}")
    print("=" * 62)
    print(summary.to_string(index=False, float_format="{:.1f}".format))
    print("=" * 62 + "\n")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    UNDERLYING     = "AAPL US Equity"
    OPTION_TYPE    = "C"            # calls; change to "P" or "both" as needed
    RATE_TICKER    = "US0003M Index"
    MIN_DTE        = 7
    MAX_DTE        = 365            # ~12 months out
    MONEYNESS_BAND = 0.30           # ±30 % around spot

    # 1. Build the IV DataFrame
    iv_df, spot = build_iv_dataframe(
        underlying=UNDERLYING,
        option_type=OPTION_TYPE,
        rate_ticker=RATE_TICKER,
        min_expiry_days=MIN_DTE,
        max_expiry_days=MAX_DTE,
        moneyness_band=MONEYNESS_BAND,
    )

    if iv_df.empty:
        print("[surface] No valid IV nodes — check Bloomberg connection and option chain.")
        return

    # 2. Print summary
    surface_summary(iv_df, underlying=UNDERLYING)

    # 3. Plot
    plot_iv_surface(iv_df, spot=spot, underlying=UNDERLYING, typ=OPTION_TYPE)

    # 4. Optionally export the pivot matrix to CSV
    pivot = pivot_surface(iv_df)
    pivot.to_csv(BASE_DIR / f"outputs/{UNDERLYING}_{OPTION_TYPE}_iv_surface.csv", index=False)
    print(f"[surface] Surface matrix saved → {UNDERLYING}_{OPTION_TYPE}_iv_surface.csv")


main()