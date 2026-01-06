import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import pytz

import matplotlib.pyplot as plt

# Databento (pip install databento)
import databento as db


# ============================================================
# CONFIG (tweaked)
# ============================================================

@dataclass
class Config:
    # ---- Databento ----
    dataset: str = "GLBX.MDP3"
    symbol: str = "NQ.v.0"
    schema: str = "ohlcv-1m"
    start: str = "2024-06-01"
    end: str = "2024-06-30"
    stype_in: str = "continuous"

    # ---- Timezone/session ----
    tz: str = "America/New_York"
    rth_open: str = "09:30"
    or_end: str = "09:45"
    rth_close: str = "16:00"
    last_entry: str = "10:30"

    # ---- Core rules ----
    max_trades_per_day: int = 2
    stop_after_first_win: bool = True

    clean_break_points: float = 3.0

    # Imbalance (FVG-style)
    require_candle2_direction: bool = False
    require_candle3_direction: bool = True

    stop_mode: str = "minmax3"

    # Stop buffer
    use_fixed_stop_buffer: bool = True
    stop_buffer_points: float = 2.0

    use_risk_buffer: bool = False
    stop_buffer_risk_frac: float = 0.25

    # RR
    stop_threshold_points: float = 30.0
    rr_small_stop: float = 2.0
    rr_big_stop: float = 1.0

    # Retest confirmation
    use_retest_confirmation: bool = True
    retest_wait_bars: int = 1
    retest_require_directional_close: bool = True

    # Continuation filter
    use_continuation_filter: bool = True
    continuation_bars: int = 5

    # Gap confirmation + hold
    use_gap_confirmation: bool = True
    gap_min_points: float = 5.0
    gap_directional: bool = True

    use_gap_hold_filter: bool = True
    gap_hold_mode: str = "or_extreme"  # "or_extreme" or "gap_fill_50"
    gap_fill_frac: float = 0.5

    # BE (structure + RR)
    use_breakeven_structure: bool = True
    pivot_lookback: int = 5

    use_breakeven_rr: bool = True
    breakeven_rr_trigger: float = 0.75

    # Fill assumptions
    same_bar_rule: str = "conservative"

    # Costs (points)
    slippage_points: float = 0.0
    commission_points_roundturn: float = 0.0

    # Output
    out_csv: str = "trades_orb_imbalance_tweaked.csv"


CFG = Config()


# ============================================================
# TIME HELPERS
# ============================================================

def parse_time_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)

def between_times(ts: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    sh, sm = parse_time_hhmm(start_hhmm)
    eh, em = parse_time_hhmm(end_hhmm)
    t = ts.timetz()
    return (t.hour, t.minute) >= (sh, sm) and (t.hour, t.minute) < (eh, em)

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


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_ohlcv_1m_from_databento(cfg: Config) -> pd.DataFrame:
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DATABENTO_API_KEY env var.")

    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset=cfg.dataset,
        schema=cfg.schema,
        symbols=cfg.symbol,
        stype_in=cfg.stype_in,
        start=cfg.start,
        end=cfg.end,
    )
    df = data.to_df()

    # Find timestamp column / reset index
    if df.index.name and "ts_" in str(df.index.name).lower():
        df = df.reset_index()
        ts_col = df.columns[0]
    else:
        ts_candidates = [c for c in df.columns if "ts_" in c.lower()]
        if ts_candidates:
            ts_col = ts_candidates[0]
        else:
            df = df.reset_index()
            ts_col = df.columns[0]

    # Normalize OHLCV
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

    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Have: {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)

    ny = pytz.timezone(cfg.tz)
    df.index = df.index.tz_convert(ny)

    return df[needed].copy()


# ============================================================
# IMBALANCE + STOP
# ============================================================

def bullish_imbalance(c1, c2, c3, cfg: Config) -> bool:
    (o1, h1, l1, c1c) = c1
    (o2, h2, l2, c2c) = c2
    (o3, h3, l3, c3c) = c3
    cond_gap = l3 > h1
    cond_dir2 = candle_is_up(o2, c2c) if cfg.require_candle2_direction else True
    cond_dir3 = candle_is_up(o3, c3c) if cfg.require_candle3_direction else True
    return cond_gap and cond_dir2 and cond_dir3

def bearish_imbalance(c1, c2, c3, cfg: Config) -> bool:
    (o1, h1, l1, c1c) = c1
    (o2, h2, l2, c2c) = c2
    (o3, h3, l3, c3c) = c3
    cond_gap = h3 < l1
    cond_dir2 = candle_is_down(o2, c2c) if cfg.require_candle2_direction else True
    cond_dir3 = candle_is_down(o3, c3c) if cfg.require_candle3_direction else True
    return cond_gap and cond_dir2 and cond_dir3

def compute_stop_base(direction: str, r1, r2, r3, cfg: Config) -> float:
    if cfg.stop_mode not in ("minmax3", "c1", "c2", "c3"):
        raise ValueError("stop_mode must be one of: minmax3, c1, c2, c3")

    if direction == "long":
        if cfg.stop_mode == "minmax3":
            return float(min(r1["low"], r2["low"], r3["low"]))
        if cfg.stop_mode == "c1":
            return float(r1["low"])
        if cfg.stop_mode == "c2":
            return float(r2["low"])
        return float(r3["low"])

    if cfg.stop_mode == "minmax3":
        return float(max(r1["high"], r2["high"], r3["high"]))
    if cfg.stop_mode == "c1":
        return float(r1["high"])
    if cfg.stop_mode == "c2":
        return float(r2["high"])
    return float(r3["high"])

def apply_stop_buffers(direction: str, entry: float, stop_base: float, cfg: Config) -> float:
    stop = float(stop_base)

    if cfg.use_fixed_stop_buffer and cfg.stop_buffer_points > 0:
        stop = stop - cfg.stop_buffer_points if direction == "long" else stop + cfg.stop_buffer_points

    if cfg.use_risk_buffer and cfg.stop_buffer_risk_frac > 0:
        risk0 = (entry - stop) if direction == "long" else (stop - entry)
        if risk0 > 0:
            extra = cfg.stop_buffer_risk_frac * risk0
            stop = stop - extra if direction == "long" else stop + extra

    return stop


# ============================================================
# BE STRUCTURE
# ============================================================

def is_pivot_high(highs: np.ndarray, i: int, lb: int) -> bool:
    center = highs[i]
    left = highs[i - lb:i]
    right = highs[i + 1:i + lb + 1]
    return np.all(center > left) and np.all(center > right)

def is_pivot_low(lows: np.ndarray, i: int, lb: int) -> bool:
    center = lows[i]
    left = lows[i - lb:i]
    right = lows[i + 1:i + lb + 1]
    return np.all(center < left) and np.all(center < right)


def simulate_trade(
    day_df: pd.DataFrame,
    entry_i: int,
    direction: str,
    entry: float,
    stop: float,
    tp: float,
    cfg: Config
) -> Tuple[int, float, str, bool, bool]:
    entry_eff = entry + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)
    be_price = entry_eff

    moved_to_be = False
    be_active = False
    pivot_be_used = False

    highs = day_df["high"].to_numpy(dtype=float)
    lows = day_df["low"].to_numpy(dtype=float)

    lb = int(cfg.pivot_lookback)
    last_pivot_level: Optional[float] = None

    risk = abs(entry_eff - stop)

    for i in range(entry_i + 1, len(day_df)):
        hi = float(day_df.iloc[i]["high"])
        lo = float(day_df.iloc[i]["low"])

        # BE by RR
        if cfg.use_breakeven_rr and not be_active and risk > 0:
            if direction == "long" and hi >= entry_eff + cfg.breakeven_rr_trigger * risk:
                be_active = True
                moved_to_be = True
            elif direction == "short" and lo <= entry_eff - cfg.breakeven_rr_trigger * risk:
                be_active = True
                moved_to_be = True

        # BE by structure (pivot break)
        if cfg.use_breakeven_structure and not be_active:
            k = i - lb
            if k - lb >= entry_i and k + lb < len(day_df):
                if direction == "long" and is_pivot_high(highs, k, lb):
                    last_pivot_level = float(highs[k])
                if direction == "short" and is_pivot_low(lows, k, lb):
                    last_pivot_level = float(lows[k])

            if last_pivot_level is not None:
                if direction == "long" and hi > last_pivot_level:
                    be_active = True
                    moved_to_be = True
                    pivot_be_used = True
                elif direction == "short" and lo < last_pivot_level:
                    be_active = True
                    moved_to_be = True
                    pivot_be_used = True

        cur_stop = be_price if be_active else stop

        hit_sl = (lo <= cur_stop) if direction == "long" else (hi >= cur_stop)
        hit_tp = (hi >= tp) if direction == "long" else (lo <= tp)

        if hit_sl and hit_tp:
            if cfg.same_bar_rule == "conservative":
                return i, cur_stop, ("BE" if be_active else "SL"), moved_to_be, pivot_be_used
            return i, tp, "TP", moved_to_be, pivot_be_used

        if hit_sl:
            return i, cur_stop, ("BE" if be_active else "SL"), moved_to_be, pivot_be_used
        if hit_tp:
            return i, tp, "TP", moved_to_be, pivot_be_used

    return len(day_df) - 1, float(day_df.iloc[-1]["close"]), "EOD", moved_to_be, pivot_be_used


# ============================================================
# BACKTEST
# ============================================================

@dataclass
class Trade:
    date: str
    direction: str
    entry_time: str
    exit_time: str
    entry: float
    stop: float
    tp: float
    exit: float
    result_r: float
    reason: str
    risk_points: float
    gap_points: float
    moved_to_be: bool
    be_by_pivot: bool


def backtest_orb_imbalance(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    trades: List[Trade] = []
    df = df.sort_index().copy()

    rth_mask = df.index.map(lambda ts: is_rth(ts, cfg))
    df_rth = df[rth_mask].copy()
    rth_close_by_day = df_rth.groupby(df_rth.index.date)["close"].last().to_dict()

    for day, day_df_all in df.groupby(df.index.date):
        day_df_rth = day_df_all[day_df_all.index.map(lambda ts: is_rth(ts, cfg))].copy()
        if len(day_df_rth) < 50:
            continue

        or_df = day_df_rth[day_df_rth.index.map(lambda ts: is_or_window(ts, cfg))]
        if len(or_df) < 5:
            continue

        or_high = float(or_df["high"].max())
        or_low = float(or_df["low"].min())
        today_open = float(or_df.iloc[0]["open"])

        prev_day = (pd.Timestamp(day) - pd.Timedelta(days=1)).date()
        prev_rth_close = rth_close_by_day.get(prev_day, None)
        gap = 0.0 if prev_rth_close is None else (today_open - float(prev_rth_close))

        scan_df = day_df_rth[day_df_rth.index.map(lambda ts: is_entry_window(ts, cfg))]
        if len(scan_df) < 5:
            continue

        pos_map: Dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(day_df_rth.index)}

        day_trades = 0
        first_trade_won = False

        for ts in scan_df.index:
            if day_trades >= cfg.max_trades_per_day:
                break
            if cfg.stop_after_first_win and first_trade_won:
                break

            i = pos_map[ts]
            if i < 2:
                continue

            close_i = float(day_df_rth.iloc[i]["close"])

            direction: Optional[str] = None
            if close_i >= (or_high + cfg.clean_break_points):
                direction = "long"
            elif close_i <= (or_low - cfg.clean_break_points):
                direction = "short"
            else:
                continue

            # Gap confirmation
            if cfg.use_gap_confirmation and prev_rth_close is not None:
                if cfg.gap_directional:
                    if direction == "long" and gap < cfg.gap_min_points:
                        continue
                    if direction == "short" and gap > -cfg.gap_min_points:
                        continue
                else:
                    if abs(gap) < cfg.gap_min_points:
                        continue

            # Continuation filter
            if cfg.use_continuation_filter:
                j_end = min(i + cfg.continuation_bars, len(day_df_rth) - 1)
                window = day_df_rth.iloc[i:j_end + 1]
                if direction == "long":
                    if float(window["high"].max()) <= float(day_df_rth.iloc[i]["high"]):
                        continue
                else:
                    if float(window["low"].min()) >= float(day_df_rth.iloc[i]["low"]):
                        continue

            # Imbalance
            r1 = day_df_rth.iloc[i - 2]
            r2 = day_df_rth.iloc[i - 1]
            r3 = day_df_rth.iloc[i]

            c1 = (float(r1["open"]), float(r1["high"]), float(r1["low"]), float(r1["close"]))
            c2 = (float(r2["open"]), float(r2["high"]), float(r2["low"]), float(r2["close"]))
            c3 = (float(r3["open"]), float(r3["high"]), float(r3["low"]), float(r3["close"]))

            ok = bullish_imbalance(c1, c2, c3, cfg) if direction == "long" else bearish_imbalance(c1, c2, c3, cfg)
            if not ok:
                continue

            # initial entry on close of imbalance candle
            entry_i = i
            entry = close_i

            stop_base = compute_stop_base(direction, r1, r2, r3, cfg)
            stop = apply_stop_buffers(direction, entry, stop_base, cfg)

            # Retest confirmation
            if cfg.use_retest_confirmation:
                wait = int(cfg.retest_wait_bars)
                if entry_i + wait >= len(day_df_rth):
                    continue
                entry_i2 = entry_i + wait
                bar = day_df_rth.iloc[entry_i2]

                imb_low = float(min(r1["low"], r2["low"], r3["low"]))
                imb_high = float(max(r1["high"], r2["high"], r3["high"]))

                if direction == "long":
                    if float(bar["low"]) < imb_low:
                        continue
                    if cfg.retest_require_directional_close and not candle_is_up(float(bar["open"]), float(bar["close"])):
                        continue
                else:
                    if float(bar["high"]) > imb_high:
                        continue
                    if cfg.retest_require_directional_close and not candle_is_down(float(bar["open"]), float(bar["close"])):
                        continue

                entry_i = entry_i2
                entry = float(bar["close"])
                stop = apply_stop_buffers(direction, entry, stop_base, cfg)

            # Gap hold filter
            if cfg.use_gap_hold_filter and prev_rth_close is not None and cfg.use_gap_confirmation:
                pre_entry = day_df_rth.iloc[:entry_i + 1]
                if cfg.gap_hold_mode == "or_extreme":
                    if direction == "long":
                        if float(pre_entry["low"].min()) < or_low:
                            continue
                    else:
                        if float(pre_entry["high"].max()) > or_high:
                            continue
                elif cfg.gap_hold_mode == "gap_fill_50" and gap != 0:
                    fill_level = today_open - cfg.gap_fill_frac * gap
                    if direction == "long":
                        if float(pre_entry["low"].min()) <= min(today_open, fill_level):
                            continue
                    else:
                        if float(pre_entry["high"].max()) >= max(today_open, fill_level):
                            continue

            risk_points = (entry - stop) if direction == "long" else (stop - entry)
            if risk_points <= 0:
                continue

            rr = cfg.rr_small_stop if risk_points < cfg.stop_threshold_points else cfg.rr_big_stop
            tp = entry + rr * risk_points if direction == "long" else entry - rr * risk_points

            exit_i, exit_px, reason, moved_to_be, be_by_pivot = simulate_trade(
                day_df=day_df_rth,
                entry_i=entry_i,
                direction=direction,
                entry=entry,
                stop=stop,
                tp=tp,
                cfg=cfg
            )

            pnl_points = (exit_px - entry) if direction == "long" else (entry - exit_px)
            pnl_points -= cfg.commission_points_roundturn
            result_r = pnl_points / risk_points

            trades.append(Trade(
                date=str(day),
                direction=direction,
                entry_time=str(day_df_rth.index[entry_i]),
                exit_time=str(day_df_rth.index[exit_i]),
                entry=float(entry),
                stop=float(stop),
                tp=float(tp),
                exit=float(exit_px),
                result_r=float(result_r),
                reason=reason,
                risk_points=float(risk_points),
                gap_points=float(gap),
                moved_to_be=bool(moved_to_be),
                be_by_pivot=bool(be_by_pivot),
            ))

            day_trades += 1
            if result_r > 0:
                first_trade_won = True

    return pd.DataFrame([t.__dict__ for t in trades])


# ============================================================
# PLOTTING (for viewer)
# ============================================================

def plot_trade(
    day_df_rth: pd.DataFrame,
    trade_row: pd.Series,
    cfg: Config,
    or_high: float,
    or_low: float,
):
    """
    Plot close + (high/low faint), entry/stop/tp, and OR box.
    trade_row needs: direction, entry_time, exit_time, entry, stop, tp, exit, reason, result_r
    """
    entry_time = pd.to_datetime(trade_row["entry_time"])
    exit_time = pd.to_datetime(trade_row["exit_time"])

    # Plot window: 09:30 to 16:00 (RTH)
    x = day_df_rth.index
    close = day_df_rth["close"].astype(float).to_numpy()
    hi = day_df_rth["high"].astype(float).to_numpy()
    lo = day_df_rth["low"].astype(float).to_numpy()

    entry = float(trade_row["entry"])
    stop = float(trade_row["stop"])
    tp = float(trade_row["tp"])

    title = f'{trade_row["date"]} | {trade_row["direction"].upper()} | {trade_row["reason"]} | R={float(trade_row["result_r"]):.2f}'
    plt.figure(figsize=(14, 6))
    plt.title(title, fontsize=14)

    plt.plot(x, close, linewidth=1.8, label="Close")
    plt.plot(x, hi, linewidth=0.8, alpha=0.25, label="High")
    plt.plot(x, lo, linewidth=0.8, alpha=0.25, label="Low")

    # OR box lines
    plt.axhline(or_high, linestyle="--", linewidth=1.0, alpha=0.4, label="OR High")
    plt.axhline(or_low, linestyle="--", linewidth=1.0, alpha=0.4, label="OR Low")

    # Entry/Stop/TP
    plt.axhline(entry, linestyle="--", linewidth=2.0, label=f"Entry {entry:.2f}")
    plt.axhline(stop, linestyle="--", linewidth=2.0, label=f"Stop {stop:.2f}")
    plt.axhline(tp, linestyle="--", linewidth=2.0, label=f"TP {tp:.2f}")

    # entry/exit markers
    if entry_time in day_df_rth.index:
        plt.scatter([entry_time], [entry], s=90, zorder=5, label="Entry")
        plt.axvline(entry_time, linewidth=2.0, alpha=0.25)
    if exit_time in day_df_rth.index:
        exit_px = float(trade_row["exit"])
        plt.scatter([exit_time], [exit_px], s=90, marker="x", zorder=6, label="Exit")
        plt.axvline(exit_time, linewidth=2.0, alpha=0.25)

    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def view_trades(csv_path: str, cfg: Config):
    trades = pd.read_csv(csv_path)
    if trades.empty:
        print(f"No trades in {csv_path}")
        return

    print(f"Loaded {len(trades)} trades from {csv_path}")
    print(trades[["date", "direction", "entry_time", "exit_time", "entry", "stop", "tp", "result_r", "reason"]].head(20))

    print("\nDownloading data for visualization...")
    df = download_ohlcv_1m_from_databento(cfg)

    # Precompute RTH closes for gap, and per-day RTH slice
    df_rth = df[df.index.map(lambda ts: is_rth(ts, cfg))].copy()

    for idx, tr in trades.iterrows():
        day = pd.to_datetime(tr["date"]).date()
        day_df_rth = df_rth[df_rth.index.date == day].copy()
        if day_df_rth.empty:
            print(f"[{idx+1}/{len(trades)}] Missing day data: {day}")
            continue

        # compute OR levels for that day
        or_df = day_df_rth[day_df_rth.index.map(lambda ts: is_or_window(ts, cfg))]
        if len(or_df) < 2:
            print(f"[{idx+1}/{len(trades)}] Missing OR slice for {day}")
            continue
        or_high = float(or_df["high"].max())
        or_low = float(or_df["low"].min())

        print(f'\n[{idx+1}/{len(trades)}] {tr["date"]} {tr["direction"].upper()} @ {float(tr["entry"]):.2f}')
        print(f'    Entry: {tr["entry_time"]}')
        print(f'    Exit : {tr["exit_time"]} ({tr["reason"]})  R={float(tr["result_r"]):.2f}')
        print("Closing each chart window will show the next trade.")

        plot_trade(day_df_rth, tr, cfg, or_high, or_low)


# ============================================================
# STATS + MAIN
# ============================================================

def compute_stats(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {"trades": 0}

    r = trades["result_r"].astype(float)
    wins = r[r > 0]
    losses = r[r <= 0]

    equity = r.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    pf = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-12 else float("inf")

    return {
        "trades": float(len(trades)),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
        "profit_factor": pf,
        "max_drawdown_r": max_dd,
        "best_r": float(r.max()),
        "worst_r": float(r.min()),
        "moved_to_be_rate": float(trades["moved_to_be"].mean()) if "moved_to_be" in trades else 0.0,
        "be_by_pivot_rate": float(trades["be_by_pivot"].mean()) if "be_by_pivot" in trades else 0.0,
    }


def run_backtest(cfg: Config) -> pd.DataFrame:
    print("=== ORB + Imbalance (Tweaked) Backtest ===")
    print(f"Dataset={cfg.dataset} Symbol={cfg.symbol} Range={cfg.start} -> {cfg.end}")
    print(f"OR: {cfg.rth_open}-{cfg.or_end} | Entries until {cfg.last_entry}")
    print(f"CleanBreak={cfg.clean_break_points} | GapConfirm={cfg.use_gap_confirmation} (min={cfg.gap_min_points}, directional={cfg.gap_directional})")
    print(f"StopMode={cfg.stop_mode} | StopBuffer(fixed)={cfg.use_fixed_stop_buffer}:{cfg.stop_buffer_points} | StopBuffer(risk)={cfg.use_risk_buffer}:{cfg.stop_buffer_risk_frac}")
    print(f"RetestConfirm={cfg.use_retest_confirmation} wait={cfg.retest_wait_bars} directional_close={cfg.retest_require_directional_close}")
    print(f"ContinuationFilter={cfg.use_continuation_filter} bars={cfg.continuation_bars}")
    print(f"BE: structure={cfg.use_breakeven_structure} pivot_lb={cfg.pivot_lookback} | rr={cfg.use_breakeven_rr} at {cfg.breakeven_rr_trigger}R")
    print(f"RR: <{cfg.stop_threshold_points} pts => {cfg.rr_small_stop}R, else {cfg.rr_big_stop}R")
    print(f"GapHold={cfg.use_gap_hold_filter} mode={cfg.gap_hold_mode}")

    df = download_ohlcv_1m_from_databento(cfg)
    print(f"\nDownloaded bars: {len(df):,}")
    print(df.head())

    trades = backtest_orb_imbalance(df, cfg)
    trades.to_csv(cfg.out_csv, index=False)
    print(f"\nSaved trades to: {cfg.out_csv}")

    stats = compute_stats(trades)
    print("\n=== STATS (R multiples) ===")
    for k, v in stats.items():
        print(f"{k:>20}: {v}")

    if not trades.empty:
        print("\nSample trades (with entry/exit times):")
        show = trades[[
            "date", "direction", "entry_time", "exit_time",
            "entry", "stop", "tp", "exit",
            "result_r", "reason"
        ]].head(20)
        print(show.to_string(index=False))

    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="View charts for trades in CSV.")
    parser.add_argument("--csv", type=str, default=CFG.out_csv, help="CSV path to view (default: out_csv).")
    args = parser.parse_args()

    if args.view:
        if not os.path.exists(args.csv):
            print(f"CSV not found: {args.csv}")
            print("Run backtest first: python orb_imbalance_backtest_improved.py")
            sys.exit(1)
        view_trades(args.csv, CFG)
    else:
        run_backtest(CFG)


if __name__ == "__main__":
    main()
