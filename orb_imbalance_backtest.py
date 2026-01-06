import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import pytz

# Optional (only used if you actually download from Databento)
import databento as db  # pip install databento


# =========================
# CONFIG
# =========================

@dataclass
class Config:
    # ---- Market / data ----
    dataset: str = "GLBX.MDP3"          # CME Globex MDP 3.0 (Databento)
    symbol: str = "NQ.v.0"              # Continuous contract front month (Databento symbology)
    schema: str = "ohlcv-1m"            # 1-minute OHLCV schema
    start: str = "2024-06-01"           # YYYY-MM-DD (reduced period for testing)
    end: str = "2024-06-30"             # YYYY-MM-DD

    # ---- Session times (NY) ----
    tz: str = "America/New_York"
    rth_open: str = "09:30"
    or_end: str = "09:45"               # Opening range is [09:30, 09:45)
    last_entry: str = "11:00"           # Stop taking new entries after this
    rth_close: str = "16:00"

    # ---- Strategy rules ----
    max_trades_per_day: int = 2
    stop_after_first_win: bool = True

    # "Imbalance" definition parameters (3-candle)
    # Bullish imbalance: low[c3] > high[c1] AND candle2 close > open (optional)
    require_candle2_direction: bool = True

    # Risk/Reward rule from the post:
    # If stop < 30 points => 2R else 1R
    rr_small_stop: float = 2.0
    rr_big_stop: float = 1.0
    stop_threshold_points: float = 30.0

    # Optional: move stop to breakeven after price moves +1R in favor
    use_breakeven: bool = True

    # Backtest fill assumptions (important!)
    # If both SL and TP touched in the same 1m bar, "conservative" assumes SL hits first.
    same_bar_rule: str = "conservative"  # "conservative" or "optimistic"

    # Costs (very rough). For NQ/MNQ you might want to set commission & slippage.
    commission_per_trade_points: float = 0.0
    slippage_points: float = 0.0


CFG = Config()


# =========================
# DATA DOWNLOAD (DATABENTO)
# =========================

def download_ohlcv_1m_from_databento(cfg: Config) -> pd.DataFrame:
    """
    Downloads 1-minute OHLCV from Databento and returns a DataFrame with:
    index: timezone-aware timestamps (America/New_York)
    columns: open, high, low, close, volume
    """
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DATABENTO_API_KEY environment variable.")

    client = db.Historical(key=api_key)

    # Databento timeseries.get_range can return OHLCV bars directly with schema ohlcv-1m.
    # Docs: schemas include OHLCV aggregate bars like 1-minute. (schema name used by Databento is ohlcv-1m)
    data = client.timeseries.get_range(
        dataset=cfg.dataset,
        schema=cfg.schema,
        symbols=cfg.symbol,
        stype_in="continuous",  # Important: specify continuous contract type
        start=cfg.start,
        end=cfg.end,
    )

    df = data.to_df()

    # For DBN data, the index is already the timestamp
    # Reset index to get timestamp as a column if needed
    if df.index.name and 'ts_' in str(df.index.name):
        df = df.reset_index()
        ts_col = df.index.name if df.index.name else df.columns[0]
    else:
        # Look for timestamp column
        ts_col_candidates = [c for c in df.columns if 'ts_' in c.lower()]
        if ts_col_candidates:
            ts_col = ts_col_candidates[0]
        else:
            # DBN format may have timestamp in index
            df = df.reset_index()
            ts_col = df.columns[0]  # Usually first column after reset

    # Normalize OHLCV field names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl in ("volume", "vol"):
            col_map[c] = "volume"

    df = df.rename(columns=col_map)

    missing = [c for c in ["open", "high", "low", "close", "volume"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected OHLCV columns: {missing}. Columns: {df.columns}")

    # Parse timestamps and localize to New York
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    df = df.set_index(ts_col)

    ny = pytz.timezone(cfg.tz)
    df.index = df.index.tz_convert(ny)

    # Keep only needed columns
    df = df[["open", "high", "low", "close", "volume"]].copy()
    return df


# =========================
# STRATEGY LOGIC
# =========================

def parse_time_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)

def between_times(ts: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    sh, sm = parse_time_hhmm(start_hhmm)
    eh, em = parse_time_hhmm(end_hhmm)
    t = ts.timetz()
    start_ok = (t.hour, t.minute) >= (sh, sm)
    end_ok = (t.hour, t.minute) < (eh, em)
    return start_ok and end_ok

def is_rth(ts: pd.Timestamp, cfg: Config) -> bool:
    return between_times(ts, cfg.rth_open, cfg.rth_close)

def is_or_window(ts: pd.Timestamp, cfg: Config) -> bool:
    return between_times(ts, cfg.rth_open, cfg.or_end)

def is_entry_window(ts: pd.Timestamp, cfg: Config) -> bool:
    return between_times(ts, cfg.or_end, cfg.last_entry)

def candle_is_up(o: float, c: float) -> bool:
    return c > o

def candle_is_down(o: float, c: float) -> bool:
    return c < o

def bullish_imbalance(c1, c2, c3, require_dir: bool) -> bool:
    # c = (open, high, low, close)
    o1,h1,l1,c1c = c1
    o2,h2,l2,c2c = c2
    o3,h3,l3,c3c = c3
    cond_gap = l3 > h1
    cond_dir = candle_is_up(o2, c2c) if require_dir else True
    return cond_gap and cond_dir

def bearish_imbalance(c1, c2, c3, require_dir: bool) -> bool:
    o1,h1,l1,c1c = c1
    o2,h2,l2,c2c = c2
    o3,h3,l3,c3c = c3
    cond_gap = h3 < l1
    cond_dir = candle_is_down(o2, c2c) if require_dir else True
    return cond_gap and cond_dir

@dataclass
class Trade:
    date: str
    direction: str  # "long" / "short"
    entry_time: str
    entry: float
    stop: float
    tp: float
    exit_time: str
    exit: float
    result_r: float
    reason: str  # "TP" / "SL" / "EOD" / "BE"
    risk_points: float


def simulate_trade_path(
    day_df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    entry: float,
    stop: float,
    tp: float,
    cfg: Config
) -> Tuple[int, float, str]:
    """
    Walk forward bar-by-bar from entry_idx+1 to end of day to see where SL/TP hits.
    Returns: (exit_bar_index, exit_price, reason)
    """
    # Apply slippage/commission as points; simplistic:
    entry_eff = entry + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)

    # Breakeven activation:
    risk = abs(entry_eff - stop)
    be_activated = False
    be_price = entry_eff

    for i in range(entry_idx + 1, len(day_df)):
        row = day_df.iloc[i]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        # Optional: activate BE after +1R
        if cfg.use_breakeven and not be_activated:
            if direction == "long" and high >= entry_eff + risk:
                be_activated = True
            elif direction == "short" and low <= entry_eff - risk:
                be_activated = True

        # Determine current effective stop (BE or original)
        cur_stop = stop
        if be_activated:
            cur_stop = be_price  # move stop to entry

        # Check hits in this bar
        hit_sl = (low <= cur_stop) if direction == "long" else (high >= cur_stop)
        hit_tp = (high >= tp) if direction == "long" else (low <= tp)

        if hit_sl and hit_tp:
            # Ambiguous within same 1m bar. Choose rule.
            if cfg.same_bar_rule == "conservative":
                return i, cur_stop, ("BE" if be_activated else "SL")
            else:
                return i, tp, "TP"

        if hit_sl:
            return i, cur_stop, ("BE" if be_activated else "SL")

        if hit_tp:
            return i, tp, "TP"

    # If no hit by end of day, exit at last close
    return len(day_df) - 1, float(day_df.iloc[-1]["close"]), "EOD"


def backtest_orb_imbalance(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Core backtest: returns trades as DataFrame
    """
    if df.index.tz is None:
        raise ValueError("DataFrame index must be timezone-aware (America/New_York).")

    # Keep only RTH rows for simplicity
    df = df[df.index.map(lambda ts: is_rth(ts, cfg))].copy()

    trades: List[Trade] = []

    for day, day_df in df.groupby(df.index.date):
        day_df = day_df.copy()
        day_df["time"] = day_df.index

        # Build opening range from 09:30 to 09:45 (exclusive)
        or_df = day_df[day_df["time"].map(lambda ts: is_or_window(ts, cfg))]
        if len(or_df) < 5:
            continue

        or_high = float(or_df["high"].max())
        or_low = float(or_df["low"].min())

        # Only look for entries after 09:45 until last_entry
        scan_df = day_df[day_df["time"].map(lambda ts: is_entry_window(ts, cfg))]
        if len(scan_df) < 10:
            continue

        day_trades = 0
        first_trade_won = False

        # We'll scan bar by bar in the entry window
        scan_idx = scan_df.index
        # map index positions to the day_df integer positions
        day_df_reset = day_df.reset_index(drop=False).rename(columns={"index": "ts"})
        # We'll use integer indexing; create a mapping from timestamp -> integer position
        pos_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(day_df["time"])}

        # We need 3-bar windows for imbalance check, so iterate over bars with enough lookback.
        for ts in scan_idx:
            if day_trades >= cfg.max_trades_per_day:
                break
            if cfg.stop_after_first_win and first_trade_won:
                break

            i = pos_map[ts]
            # Need c1,c2,c3 ending at i (i is candle 3)
            if i < 2:
                continue

            # Breakout condition must have happened before/at imbalance completion.
            close_i = float(day_df.iloc[i]["close"])

            # Determine direction based on close relative to OR
            direction: Optional[str] = None
            if close_i > or_high:
                direction = "long"
            elif close_i < or_low:
                direction = "short"
            else:
                continue  # no close outside OR -> no setup

            # Define c1,c2,c3 for imbalance
            r1 = day_df.iloc[i - 2]
            r2 = day_df.iloc[i - 1]
            r3 = day_df.iloc[i]

            c1 = (float(r1["open"]), float(r1["high"]), float(r1["low"]), float(r1["close"]))
            c2 = (float(r2["open"]), float(r2["high"]), float(r2["low"]), float(r2["close"]))
            c3 = (float(r3["open"]), float(r3["high"]), float(r3["low"]), float(r3["close"]))

            ok = False
            if direction == "long":
                ok = bullish_imbalance(c1, c2, c3, cfg.require_candle2_direction)
            else:
                ok = bearish_imbalance(c1, c2, c3, cfg.require_candle2_direction)

            if not ok:
                continue

            # Entry at close of candle 3
            entry = close_i

            # Stop at imbalance low/high
            if direction == "long":
                stop = min(c1[2], c2[2], c3[2])  # low of 3 candles (more robust than only c3)
                risk_points = entry - stop
            else:
                stop = max(c1[1], c2[1], c3[1])  # high of 3 candles
                risk_points = stop - entry

            if risk_points <= 0:
                continue

            rr = cfg.rr_small_stop if risk_points < cfg.stop_threshold_points else cfg.rr_big_stop
            tp = entry + rr * risk_points if direction == "long" else entry - rr * risk_points

            # Simulate forward
            exit_i, exit_px, reason = simulate_trade_path(
                day_df=day_df,
                entry_idx=i,
                direction=direction,
                entry=entry,
                stop=stop,
                tp=tp,
                cfg=cfg
            )

            # Apply simplistic commission in points to result
            # (Deduct commission on both entry+exit)
            commission_pts = cfg.commission_per_trade_points

            # R multiple
            pnl_points = (exit_px - entry) if direction == "long" else (entry - exit_px)
            pnl_points -= commission_pts
            result_r = pnl_points / risk_points

            t = Trade(
                date=str(day),
                direction=direction,
                entry_time=str(day_df.iloc[i]["time"]),
                entry=float(entry),
                stop=float(stop),
                tp=float(tp),
                exit_time=str(day_df.iloc[exit_i]["time"]),
                exit=float(exit_px),
                result_r=float(result_r),
                reason=reason,
                risk_points=float(risk_points),
            )
            trades.append(t)

            day_trades += 1
            if result_r > 0:
                first_trade_won = True

        # end day loop

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    return trades_df


# =========================
# METRICS
# =========================

def compute_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"trades": 0}

    r = trades["result_r"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]

    equity = r.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min() if len(dd) else 0.0

    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 1e-12 else np.inf

    stats = {
        "trades": float(len(trades)),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown_r": float(max_dd),
        "best_r": float(r.max()),
        "worst_r": float(r.min()),
    }
    return stats


# =========================
# MAIN
# =========================

def main():
    print("=== ORB + Imbalance Backtest ===")
    print(f"Dataset={CFG.dataset} Symbol={CFG.symbol} Schema={CFG.schema} Range={CFG.start} -> {CFG.end}")

    # 1) Download data
    df = download_ohlcv_1m_from_databento(CFG)
    print(f"Downloaded bars: {len(df):,}")
    print(df.head())

    # 2) Backtest
    trades = backtest_orb_imbalance(df, CFG)

    # 3) Save results
    out_trades = "trades_orb_imbalance.csv"
    trades.to_csv(out_trades, index=False)
    print(f"Saved trades to: {out_trades}")

    # 4) Stats
    stats = compute_stats(trades)
    print("\n=== STATS (R-multiples) ===")
    for k, v in stats.items():
        print(f"{k:>16}: {v}")

    if not trades.empty:
        print("\nSample trades:")
        print(trades.head(10))


if __name__ == "__main__":
    main()
