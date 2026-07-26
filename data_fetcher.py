"""
Data Fetching Module for PCR Dashboard
=======================================
Handles:
  - NSE F&O Bhav Copy fetching via nselib
  - Nifty 50 price data via yfinance
  - Local CSV caching for fast reloads
  - Realistic synthetic data generation as fallback
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Directory Setup ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Column Mappings (nselib version compatibility) ───────────────
COLUMN_MAPS = {
    'new': {
        'symbol': 'TckrSymb',
        'instrument': 'FinInstrmTp',
        'expiry': 'XpryDt',
        'strike': 'StrkPric',
        'option_type': 'OptnTp',
        'open_interest': 'OpnIntrst',
        'chg_oi': 'ChngInOpnIntrst',
        'volume': 'TtlTradgVol',
    },
    'old': {
        'symbol': 'SYMBOL',
        'instrument': 'INSTRUMENT',
        'expiry': 'EXPIRY_DT',
        'strike': 'STRIKE_PR',
        'option_type': 'OPTION_TYP',
        'open_interest': 'OPEN_INT',
        'chg_oi': 'CHG_IN_OI',
        'volume': 'CONTRACTS',
    }
}

OPTION_INSTRUMENT_VALUES = {
    'new': 'IDO',      # Index Options (new UDiFF bhav copy schema)
    'old': 'OPTIDX',   # Index Options (legacy schema)
}

def _detect_column_format(df):
    """Detect which column format the bhav copy DataFrame uses."""
    cols = set(df.columns)
    if 'TckrSymb' in cols:
        return 'new'
    elif 'SYMBOL' in cols:
        return 'old'
    # Fallback: case-insensitive match
    cols_upper = {c.upper() for c in df.columns}
    return 'new' if 'TCKRSYMB' in cols_upper else 'old'


def _get_col(df, fmt, key):
    """Resolve column name from format mapping, with case-insensitive fallback."""
    col_name = COLUMN_MAPS[fmt][key]
    if col_name in df.columns:
        return col_name
    for c in df.columns:
        if c.upper() == col_name.upper():
            return c
    return None


# ── Single Day Fetch ─────────────────────────────────────────────

def fetch_single_day_pcr(date, retry_count=2):
    """
    Fetch F&O bhav copy for one trading day and compute Nifty PCR.
    
    Returns:
        dict with keys: date, put_oi, call_oi, pcr, put_vol, call_vol, vol_pcr
        None if fetch fails (holiday, API error, etc.)
    """
    date_str = date.strftime('%d-%m-%Y')

    try:
        # pyrefly: ignore [missing-import]
        from nselib import derivatives
    except ImportError:
        return None

    for attempt in range(retry_count):
        try:
            bhav = derivatives.fno_bhav_copy(trade_date=date_str)

            if bhav is None or (hasattr(bhav, 'empty') and bhav.empty):
                return None

            fmt = _detect_column_format(bhav)
            sym_col = _get_col(bhav, fmt, 'symbol')
            inst_col = _get_col(bhav, fmt, 'instrument')
            opt_col = _get_col(bhav, fmt, 'option_type')
            oi_col = _get_col(bhav, fmt, 'open_interest')
            vol_col = _get_col(bhav, fmt, 'volume')

            if not all([sym_col, opt_col, oi_col]):
                return None

            # ── Filter for NIFTY Index Options ──

            # mask = bhav[sym_col].astype(str).str.upper() == 'NIFTY'
            # if inst_col:
            #     mask = mask & bhav[inst_col].astype(str).str.contains('OPT', case=False, na=False)
            # nifty_opts = bhav[mask].copy()

            mask = bhav[sym_col].astype(str).str.upper() == 'NIFTY'
            if inst_col:
                target_val = OPTION_INSTRUMENT_VALUES[fmt]
                mask = mask & (bhav[inst_col].astype(str).str.upper() == target_val)
            nifty_opts = bhav[mask].copy()

            if nifty_opts.empty:
                return None

            nifty_opts[oi_col] = pd.to_numeric(nifty_opts[oi_col], errors='coerce').fillna(0)

            calls = nifty_opts[nifty_opts[opt_col].astype(str).str.upper() == 'CE']
            puts = nifty_opts[nifty_opts[opt_col].astype(str).str.upper() == 'PE']

            call_oi = calls[oi_col].sum()
            put_oi = puts[oi_col].sum()
            pcr_oi = put_oi / call_oi if call_oi > 0 else None

            result = {
                'date': date,
                'put_oi': int(put_oi),
                'call_oi': int(call_oi),
                'pcr': round(pcr_oi, 4) if pcr_oi is not None else None,
            }

            # Volume-based PCR
            if vol_col and vol_col in nifty_opts.columns:
                nifty_opts[vol_col] = pd.to_numeric(nifty_opts[vol_col], errors='coerce').fillna(0)
                call_vol = calls[vol_col].sum()
                put_vol = puts[vol_col].sum()
                vol_pcr = put_vol / call_vol if call_vol > 0 else None
                result['put_vol'] = int(put_vol)
                result['call_vol'] = int(call_vol)
                result['vol_pcr'] = round(vol_pcr, 4) if vol_pcr is not None else None

            return result

        except Exception as e:
            print(f"[PCR fetch error] {date_str}: {type(e).__name__}: {e}")
            if attempt < retry_count - 1:
                time.sleep(2)
            continue
        except ImportError:
            print("[PCR fetch error] nselib not installed — run: pip install nselib")
            return None

    return None


# ── Historical PCR Fetch ─────────────────────────────────────────

def fetch_pcr_history(start_date, end_date, progress_callback=None):
    """
    Fetch daily PCR data for a date range.
    Uses local cache to avoid redundant API calls.
    
    Args:
        start_date: Start date (str or datetime)
        end_date: End date (str or datetime)
        progress_callback: Optional fn(progress_pct, message_str)
    
    Returns:
        DataFrame with columns: date, put_oi, call_oi, pcr [, vol_pcr, ...]
    """
    cache_file = os.path.join(CACHE_DIR, 'pcr_history.csv')

    # Load existing cache
    cached_df = pd.DataFrame()
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=['date'])
        except Exception:
            cached_df = pd.DataFrame()

    # Business days in range
    all_days = pd.bdate_range(start=start_date, end=end_date)

    # Find missing days
    if not cached_df.empty:
        cached_dates = set(cached_df['date'].dt.date)
        days_to_fetch = [d for d in all_days if d.date() not in cached_dates]
    else:
        days_to_fetch = list(all_days)

    # Exclude known non-trading days (holidays we already failed on)
    failed_cache = os.path.join(CACHE_DIR, 'failed_dates.txt')
    failed_dates = set()
    if os.path.exists(failed_cache):
        with open(failed_cache, 'r') as f:
            content = f.read().strip()
            if content:
                failed_dates = set(content.split('\n'))

    days_to_fetch = [d for d in days_to_fetch if d.strftime('%Y-%m-%d') not in failed_dates]
    total = len(days_to_fetch)

    new_records = []
    for i, day in enumerate(days_to_fetch):
        if progress_callback:
            progress_callback(
                (i + 1) / total if total > 0 else 1.0,
                f"Fetching {day.strftime('%d-%b-%Y')}... ({i + 1}/{total})"
            )

        result = fetch_single_day_pcr(day)
        if result and result['pcr'] is not None:
            new_records.append(result)
        elif day.date() < (datetime.today().date() - timedelta(days=1)):
            failed_dates.add(day.strftime('%Y-%m-%d'))  # only blacklist old, confirmed-no-data days

        time.sleep(0.7)  # Rate-limit NSE requests

    # Persist failed dates
    with open(failed_cache, 'w') as f:
        f.write('\n'.join(sorted(failed_dates)))

    # Merge new data with cache
    if new_records:
        new_df = pd.DataFrame(new_records)
        combined = pd.concat([cached_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        combined.to_csv(cache_file, index=False)
    else:
        combined = cached_df

    if combined.empty:
        return pd.DataFrame()

    # Filter to requested range
    mask = (combined['date'] >= pd.Timestamp(start_date)) & (combined['date'] <= pd.Timestamp(end_date))
    return combined[mask].sort_values('date').reset_index(drop=True)


# ── Nifty Price Data ─────────────────────────────────────────────

def fetch_nifty_prices(start_date, end_date):
    """
    Fetch Nifty 50 closing prices using yfinance.
    Falls back to cached CSV if yfinance is unavailable.
    """
    cache_file = os.path.join(CACHE_DIR, 'nifty_prices.csv')

    try:
        import yfinance as yf

        # Small buffer around dates
        start = (pd.Timestamp(start_date) - timedelta(days=7)).strftime('%Y-%m-%d')
        end = (pd.Timestamp(end_date) + timedelta(days=7)).strftime('%Y-%m-%d')

        nifty = yf.download('^NSEI', start=start, end=end, progress=False)

        if nifty.empty:
            raise ValueError("Empty response from yfinance")

        # Flatten multi-level columns if present
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)

        nifty = nifty.reset_index()
        rename_map = {
            'Date': 'date', 'Close': 'nifty_close',
            'Open': 'nifty_open', 'High': 'nifty_high',
            'Low': 'nifty_low', 'Volume': 'nifty_volume',
        }
        nifty = nifty.rename(columns=rename_map)

        keep = [c for c in ['date', 'nifty_close', 'nifty_open', 'nifty_high', 'nifty_low'] if c in nifty.columns]
        nifty = nifty[keep]
        nifty['date'] = pd.to_datetime(nifty['date'])
        nifty.to_csv(cache_file, index=False)
        return nifty

    except Exception:
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, parse_dates=['date'])
        return pd.DataFrame()


# ── Synthetic Data (Fallback / Demo) ─────────────────────────────

def generate_synthetic_data(start_date, end_date):
    """
    Generate realistic synthetic PCR + Nifty data for demo/testing.
    Uses Ornstein-Uhlenbeck mean-reversion for PCR and
    geometric Brownian motion for Nifty with PCR correlation.
    """
    np.random.seed(42)
    trading_days = pd.bdate_range(start=start_date, end=end_date)
    n = len(trading_days)

    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    # ── PCR: mean-reverting with regime shifts ──
    pcr = np.zeros(n)
    pcr[0] = 0.95
    mean_level = 1.0
    reversion = 0.06
    vol = 0.07

    for i in range(1, n):
        # 2% daily chance of regime shift
        if np.random.random() < 0.02:
            mean_level = np.random.choice([0.62, 0.75, 0.95, 1.10, 1.35, 1.55])
        drift = reversion * (mean_level - pcr[i - 1])
        shock = vol * np.random.randn()
        pcr[i] = np.clip(pcr[i - 1] + drift + shock, 0.35, 2.2)

    # ── Nifty: GBM with slight inverse PCR correlation ──
    nifty = np.zeros(n)
    nifty[0] = 23500
    for i in range(1, n):
        pcr_effect = -(pcr[i] - 1.0) * 0.0015
        daily_ret = 0.0004 + pcr_effect + 0.011 * np.random.randn()
        nifty[i] = nifty[i - 1] * (1 + daily_ret)

    # ── Build DataFrames ──
    put_oi = np.random.randint(800_000, 3_000_000, n)
    call_oi = (put_oi / pcr).astype(int)

    pcr_df = pd.DataFrame({
        'date': trading_days,
        'pcr': np.round(pcr, 4),
        'put_oi': put_oi,
        'call_oi': call_oi,
    })

    nifty_high = nifty * (1 + np.abs(np.random.randn(n) * 0.004))
    nifty_low = nifty * (1 - np.abs(np.random.randn(n) * 0.004))

    nifty_df = pd.DataFrame({
        'date': trading_days,
        'nifty_close': np.round(nifty, 2),
        'nifty_high': np.round(nifty_high, 2),
        'nifty_low': np.round(nifty_low, 2),
        'nifty_open': np.round(nifty * (1 + np.random.randn(n) * 0.002), 2),
    })

    return pcr_df, nifty_df
