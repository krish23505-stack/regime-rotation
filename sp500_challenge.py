"""
S&P 500 challenge: can a rules-based ETF rotation beat SPY on years it never saw?

How the test works (so the result can be trusted):
  1. 12 strategy variants are defined up front (momentum style x holdings x trend filter).
  2. Walk-forward: for each year from FIRST_TEST_YEAR on, the bot picks the variant that had the
     best Sharpe ratio using ONLY earlier years, then trades that variant for the year.
     Nothing is ever chosen by looking at the year being tested.
  3. Pass/fail criteria are fixed below BEFORE seeing results:
        - out-of-sample CAGR is higher than SPY
        - out-of-sample Sharpe is higher than SPY
        - alpha vs SPY has a t-stat of at least 2 (i.e. not just luck or extra market risk)
  4. Free Yahoo Finance data (dividend-adjusted), about 2004 to today, includes 2008.

Run:  python sp500_challenge.py
"""
VERSION = "sp500_challenge v1"

import time

import numpy as np
import pandas as pd

# ==========================================================
# CONFIG
# ==========================================================
RISK = [
    "VTI", "VXUS", "QQQ", "IWM", "VTV", "VUG",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU",
    "VNQ", "EFA", "EEM", "GLD", "TLT", "IEF",
]
CASH = "BIL"
CASH_PROXY = "SHY"          # stands in for BIL before 2007
BENCH = "SPY"               # the S&P 500 benchmark and the trend-filter symbol
EXTRA = ["SHY", "SPY", "VEU"]
DATA_START = "2003-01-01"

COST_BPS = 5
SKIP_DAYS = 21
TREND_DAYS = 200
FIRST_TEST_YEAR = 2016
MIN_TRAIN_YEARS = 3

# Pass/fail rules, fixed in advance
REQUIRED_ALPHA_T = 2.0


# ==========================================================
# DATA
# ==========================================================
def download_prices() -> pd.DataFrame:
    import yfinance as yf

    tickers = sorted(set(RISK + [CASH] + EXTRA))
    last_error = None
    for attempt in range(1, 4):
        try:
            raw = yf.download(tickers, start=DATA_START, auto_adjust=True,
                              progress=False, group_by="column", threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"] if "Close" in raw.columns.get_level_values(0) \
                    else raw.xs("Close", axis=1, level=1)
            else:
                close = raw
            close = close.dropna(how="all")
            if len(close) > 1000 and all(t in close.columns for t in tickers):
                index = pd.DatetimeIndex(close.index)
                if index.tz is not None:
                    index = index.tz_localize(None)
                close.index = index.normalize()
                return close.sort_index()
            last_error = f"only {len(close)} rows / columns {list(close.columns)}"
        except Exception as error:
            last_error = repr(error)
        print(f"Download attempt {attempt} failed: {last_error}")
        time.sleep(10 * attempt)
    raise SystemExit(f"Could not download price data: {last_error}")


def splice(primary: pd.Series, proxy: pd.Series) -> pd.Series:
    """Use primary's returns where it exists, otherwise the proxy's."""
    primary_ret = primary.pct_change()
    returns = primary_ret.where(primary_ret.notna(), proxy.pct_change())
    first = returns.first_valid_index()
    out = (1 + returns.fillna(0.0)).cumprod() * 100
    out[out.index < first] = np.nan
    return out


def build_close(raw: pd.DataFrame) -> pd.DataFrame:
    close = raw.copy()
    close[CASH] = splice(raw[CASH], raw[CASH_PROXY])
    close["VXUS"] = splice(raw["VXUS"], raw["VEU"])
    return close


# ==========================================================
# STRATEGY
# ==========================================================
def momentum_frame(close: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "blend":
        parts = [close / close.shift(n) - 1 for n in (21, 63, 126, 252)]
        return sum(parts) / len(parts)
    return close.shift(SKIP_DAYS) / close.shift(252) - 1


def run_variant(close: pd.DataFrame, mode: str, top_n: int, use_trend: bool) -> pd.Series:
    returns = close.pct_change().fillna(0.0)
    mom = momentum_frame(close, mode)
    sma = close[BENCH].rolling(TREND_DAYS).mean()
    month_ends = close.groupby([close.index.year, close.index.month]).tail(1).index
    cols = RISK + [CASH]

    signals = {}
    for date in month_ends:
        if pd.isna(sma.loc[date]) or mom.loc[date, RISK].isna().all():
            continue
        trend_ok = (close[BENCH].loc[date] > sma.loc[date]) if use_trend else True
        weights = {s: 0.0 for s in cols}
        if trend_ok:
            candidates = mom.loc[date, RISK].dropna()
            candidates = candidates[candidates > 0].sort_values(ascending=False).head(top_n)
            for symbol in candidates.index:
                weights[symbol] = 1.0 / top_n
        weights[CASH] = 1.0 - sum(weights.values())
        signals[date] = weights

    # signal at month-end close -> trade next close -> earns returns after that
    W = pd.DataFrame(signals).T.reindex(close.index).ffill().shift(2).dropna()
    gross = (W * returns[cols].loc[W.index]).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()
    return gross - turnover * COST_BPS / 10_000


def variant_grid():
    grid = []
    for mode in ("12-1", "blend"):
        for top_n in (3, 5, 8):
            for trend in (True, False):
                grid.append((mode, top_n, trend))
    return grid


def variant_name(mode, top_n, trend) -> str:
    return f"{mode} | top{top_n} | {'trend' if trend else 'no-trend'}"


def walk_forward(variants: pd.DataFrame):
    picks, pieces = {}, []
    first_date = variants.index[0]
    for year in sorted(set(variants.index.year)):
        if year < FIRST_TEST_YEAR:
            continue
        past = variants[variants.index.year < year]
        if len(past) < 252 * MIN_TRAIN_YEARS:
            continue
        sharpe = past.mean() / past.std()
        best = sharpe.idxmax()
        picks[year] = best
        pieces.append(variants.loc[variants.index.year == year, best])
    return pd.concat(pieces), picks


# ==========================================================
# STATS
# ==========================================================
def metrics(r: pd.Series) -> dict:
    equity = (1 + r).cumprod()
    years = len(r) / 252
    std = r.std()
    return {
        "CAGR %": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "Vol %": std * np.sqrt(252) * 100,
        "Sharpe": (r.mean() / std * np.sqrt(252)) if std > 0 else float("nan"),
        "Max DD %": (equity / equity.cummax() - 1).min() * 100,
        "Total %": (equity.iloc[-1] - 1) * 100,
    }


def alpha_beta(strategy: pd.Series, market: pd.Series):
    """Regression of strategy daily returns on SPY daily returns."""
    y, x = strategy.values, market.values
    n = len(y)
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha_daily = y.mean() - beta * x.mean()
    resid = y - alpha_daily - beta * x
    sigma = resid.std(ddof=2)
    sxx = ((x - x.mean()) ** 2).sum()
    se_alpha = sigma * np.sqrt(1 / n + x.mean() ** 2 / sxx)
    return alpha_daily * 252 * 100, beta, alpha_daily / se_alpha


def window_return(r: pd.Series, start: str, end: str) -> float:
    part = r.loc[start:end]
    return ((1 + part).prod() - 1) * 100 if len(part) else float("nan")


# ==========================================================
# REPORT
# ==========================================================
def report(close: pd.DataFrame) -> None:
    spy = close[BENCH].pct_change().fillna(0.0)
    ref = sum(close[s].pct_change().fillna(0.0) * w for s, w in (("VTI", 0.6), ("VXUS", 0.4)))

    grid = variant_grid()
    series = {variant_name(*g): run_variant(close, *g) for g in grid}
    variants = pd.DataFrame(series).dropna()
    print(f"Variants tested: {len(grid)}   History: {variants.index[0].date()} to {variants.index[-1].date()}\n")

    wf, picks = walk_forward(variants)
    spy_t, ref_t = spy.loc[wf.index], ref.loc[wf.index]

    print(f"WALK-FORWARD RESULT ({wf.index[0].date()} to {wf.index[-1].date()}, every year picked "
          f"using only earlier years)\n")
    table = pd.DataFrame({
        "Strategy (walk-forward)": metrics(wf),
        "S&P 500 (SPY)": metrics(spy_t),
        "VTI/VXUS 60/40": metrics(ref_t),
    }).round(2)
    print(table.to_string())

    alpha, beta, t_stat = alpha_beta(wf, spy_t)
    print(f"\nvs SPY: alpha {alpha:+.2f}% per year | beta {beta:.2f} | alpha t-stat {t_stat:.2f}")

    ms, mb = metrics(wf), metrics(spy_t)
    checks = [
        (f"Higher CAGR than SPY ({ms['CAGR %']:.2f}% vs {mb['CAGR %']:.2f}%)", ms["CAGR %"] > mb["CAGR %"]),
        (f"Higher Sharpe than SPY ({ms['Sharpe']:.2f} vs {mb['Sharpe']:.2f})", ms["Sharpe"] > mb["Sharpe"]),
        (f"Alpha t-stat at least {REQUIRED_ALPHA_T:.0f} (got {t_stat:.2f})", t_stat >= REQUIRED_ALPHA_T),
    ]
    print("\nPASS/FAIL (criteria set before seeing results):")
    for text, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    print(f"  Smaller worst drop than SPY: {ms['Max DD %']:.1f}% vs {mb['Max DD %']:.1f}% "
          f"({'yes' if ms['Max DD %'] > mb['Max DD %'] else 'no'}) (not part of the pass rule)")
    print("\nOVERALL: " + ("BEATS the S&P 500 on all three tests." if all(ok for _, ok in checks)
                           else "Does NOT clear all three tests."))

    yearly = pd.DataFrame({
        "Strategy %": (1 + wf).groupby(wf.index.year).prod() - 1,
        "SPY %": (1 + spy_t).groupby(spy_t.index.year).prod() - 1,
    }) * 100
    yearly["Diff %"] = yearly["Strategy %"] - yearly["SPY %"]
    yearly["Variant used"] = pd.Series(picks)
    print("\nYear by year:\n")
    print(yearly.round(1).to_string())
    print(f"\nYears beating SPY: {(yearly['Diff %'] > 0).sum()} of {len(yearly)}")

    print("\nStress periods (using the fixed 'blend | top5 | trend' variant):")
    stress = variants["blend | top5 | trend"]
    for start, end, label in (("2007-10-09", "2009-03-09", "2008 financial crisis"),
                              ("2020-02-19", "2020-03-23", "COVID crash"),
                              ("2022-01-03", "2022-10-12", "2022 bear market")):
        print(f"  {label:<24} strategy {window_return(stress, start, end):7.1f}%   "
              f"SPY {window_return(spy, start, end):7.1f}%")

    print("\nEvery variant, full history vs SPY (for transparency only; picking the best one here "
          "would be cheating):\n")
    full = {}
    spy_full = spy.loc[variants.index]
    for name in variants.columns:
        m = metrics(variants[name])
        full[name] = {"CAGR %": m["CAGR %"], "Sharpe": m["Sharpe"], "Max DD %": m["Max DD %"]}
    full["S&P 500 (SPY)"] = {k: metrics(spy_full)[k] for k in ("CAGR %", "Sharpe", "Max DD %")}
    print(pd.DataFrame(full).T.round(2).to_string())

    pd.DataFrame({"strategy": (1 + wf).cumprod(), "spy": (1 + spy_t).cumprod()}).to_csv(
        "sp500_challenge_equity.csv")
    print("\nSaved sp500_challenge_equity.csv")
    print("\nCAVEATS: one historical path, not a forecast. ETF history before 2007 uses a Treasury "
          "fund as cash. Costs are a flat 5 bps per unit traded. Returns are before taxes.")


def main() -> None:
    print(VERSION)
    print("Downloading price history from Yahoo Finance...")
    raw = download_prices()
    print(f"Got {len(raw)} trading days: {raw.index[0].date()} to {raw.index[-1].date()}\n")
    report(build_close(raw))


if __name__ == "__main__":
    main()
# END OF FILE
