"""
PCR Calculation & Signal Analysis Module
=========================================
Handles:
  - Moving averages & Z-scores on PCR series
  - Signal detection at user-defined PCR threshold zones
  - Forward return computation for strategy backtesting
  - Strategy performance statistics
"""

import pandas as pd
import numpy as np


# ── Moving Averages & Smoothing ──────────────────────────────────

def add_moving_averages(df, periods=None):
    """
    Add rolling moving averages of PCR to the DataFrame.
    
    Args:
        df: DataFrame with 'pcr' column
        periods: List of MA window sizes (default: [5, 21])
    
    Returns:
        DataFrame with added pcr_ma{N} columns
    """
    if periods is None:
        periods = [5, 21]

    result = df.copy()
    for p in periods:
        result[f'pcr_ma{p}'] = (
            result['pcr']
            .rolling(window=p, min_periods=max(1, p // 2))
            .mean()
            .round(4)
        )
    return result


def add_z_scores(df, window=63):
    """
    Add Z-score of PCR for statistical extreme detection.
    Z > +2 or Z < -2 flags statistically significant deviations.
    
    Args:
        df: DataFrame with 'pcr' column
        window: Rolling window for mean/std computation (default: 63 ~3 months)
    """
    result = df.copy()
    rolling_mean = result['pcr'].rolling(window=window, min_periods=20).mean()
    rolling_std = result['pcr'].rolling(window=window, min_periods=20).std()
    result['pcr_zscore'] = ((result['pcr'] - rolling_mean) / rolling_std).round(4)
    return result


def add_bollinger_bands(df, window=21, num_std=2):
    """
    Add Bollinger Bands around PCR for dynamic threshold detection.
    """
    result = df.copy()
    rolling_mean = result['pcr'].rolling(window=window, min_periods=10).mean()
    rolling_std = result['pcr'].rolling(window=window, min_periods=10).std()
    result['pcr_bb_upper'] = (rolling_mean + num_std * rolling_std).round(4)
    result['pcr_bb_lower'] = (rolling_mean - num_std * rolling_std).round(4)
    result['pcr_bb_mid'] = rolling_mean.round(4)
    return result


# ── Signal Detection ─────────────────────────────────────────────

def find_signals(df, low_range=(0.6, 0.7), high_range=(1.5, 1.6),
                 use_smoothed=False, ma_period=5, cooldown=5):
    """
    Find dates where PCR enters trigger zones.
    
    Args:
        df: DataFrame with 'pcr' and optionally 'pcr_ma{N}' columns
        low_range: Tuple (min, max) for low PCR zone — your buy signal
        high_range: Tuple (min, max) for high PCR zone — your sell signal
        use_smoothed: If True, use MA-smoothed PCR for signal detection
        ma_period: Which MA period to use if smoothed
        cooldown: Minimum trading days between consecutive same-type signals
    
    Returns:
        DataFrame with columns: signal_date, signal_type, pcr_value,
                                signal_idx, target_date
    """
    pcr_col = f'pcr_ma{ma_period}' if (use_smoothed and f'pcr_ma{ma_period}' in df.columns) else 'pcr'

    signals = []
    last_low_idx = -cooldown - 1
    last_high_idx = -cooldown - 1

    for idx in range(len(df)):
        row = df.iloc[idx]
        pcr_val = row[pcr_col]

        if pd.isna(pcr_val):
            continue

        # ── Low PCR Zone (User's bullish signal) ──
        if low_range[0] <= pcr_val <= low_range[1]:
            if (idx - last_low_idx) >= cooldown:
                signals.append({
                    'signal_date': row['date'],
                    'signal_type': 'low_pcr',
                    'pcr_value': round(pcr_val, 4),
                    'signal_idx': idx,
                })
                last_low_idx = idx

        # ── High PCR Zone (User's bearish / profit-booking signal) ──
        if high_range[0] <= pcr_val <= high_range[1]:
            if (idx - last_high_idx) >= cooldown:
                signals.append({
                    'signal_date': row['date'],
                    'signal_type': 'high_pcr',
                    'pcr_value': round(pcr_val, 4),
                    'signal_idx': idx,
                })
                last_high_idx = idx

    if not signals:
        return pd.DataFrame(columns=[
            'signal_date', 'signal_type', 'pcr_value', 'signal_idx', 'target_date'
        ])

    signal_df = pd.DataFrame(signals)

    # Compute target date: 10 trading days (~2 calendar weeks) after signal
    dates = df['date'].values
    signal_df['target_date'] = signal_df['signal_idx'].apply(
        lambda si: dates[min(si + 10, len(dates) - 1)]
    )

    return signal_df


# ── Forward Returns ──────────────────────────────────────────────

def compute_forward_returns(pcr_df, nifty_df, signals_df, periods=None):
    """
    Compute Nifty forward returns after each signal to validate strategy.
    
    Args:
        pcr_df: PCR DataFrame (for date alignment)
        nifty_df: DataFrame with 'date' and 'nifty_close'
        signals_df: Output from find_signals()
        periods: List of forward-looking periods in trading days (default: [5, 10, 15])
    
    Returns:
        signals_df with added return_{N}d and nifty_at_signal columns
    """
    if periods is None:
        periods = [5, 10, 15]

    if signals_df.empty or nifty_df.empty:
        return signals_df

    nifty_sorted = nifty_df.sort_values('date').reset_index(drop=True)
    result = signals_df.copy()

    nifty_prices_at_signal = []

    for period in periods:
        returns = []
        for row_idx, sig in signals_df.iterrows():
            sig_date = pd.Timestamp(sig['signal_date'])

            # Find closest Nifty close on or before signal date
            nifty_before = nifty_sorted[nifty_sorted['date'] <= sig_date]
            if nifty_before.empty:
                returns.append(None)
                if period == periods[0]:
                    nifty_prices_at_signal.append(None)
                continue

            entry_price = nifty_before.iloc[-1]['nifty_close']
            if period == periods[0]:
                nifty_prices_at_signal.append(round(entry_price, 2))

            # Find Nifty close after N trading days
            future = nifty_sorted[nifty_sorted['date'] > sig_date]
            if len(future) >= period:
                exit_price = future.iloc[period - 1]['nifty_close']
                ret = ((exit_price - entry_price) / entry_price) * 100
                returns.append(round(ret, 2))
            else:
                returns.append(None)

        result[f'return_{period}d'] = returns

    result['nifty_at_signal'] = nifty_prices_at_signal
    return result


# ── Strategy Performance Statistics ──────────────────────────────

def compute_strategy_stats(signals_with_returns, periods=None):
    """
    Compute aggregate strategy performance statistics.
    
    Returns:
        dict keyed by signal_type, each containing:
          total_signals, {period}d_avg_return, {period}d_win_rate,
          {period}d_max_gain, {period}d_max_loss
    """
    if periods is None:
        periods = [5, 10, 15]

    stats = {}

    for signal_type in ['low_pcr', 'high_pcr']:
        subset = signals_with_returns[signals_with_returns['signal_type'] == signal_type]

        if subset.empty:
            stats[signal_type] = {'total_signals': 0}
            continue

        type_stats = {'total_signals': len(subset)}

        for period in periods:
            col = f'return_{period}d'
            if col not in subset.columns:
                continue

            valid = subset[col].dropna()
            if len(valid) == 0:
                continue

            # Win logic:
            #   low_pcr  → user expects market UP   → win = positive return
            #   high_pcr → user expects profit-booking/DOWN → win = negative return
            if signal_type == 'low_pcr':
                wins = (valid > 0).sum()
            else:
                wins = (valid < 0).sum()

            win_rate = (wins / len(valid)) * 100

            type_stats[f'{period}d_avg_return'] = round(valid.mean(), 2)
            type_stats[f'{period}d_median_return'] = round(valid.median(), 2)
            type_stats[f'{period}d_win_rate'] = round(win_rate, 1)
            type_stats[f'{period}d_max_gain'] = round(valid.max(), 2)
            type_stats[f'{period}d_max_loss'] = round(valid.min(), 2)
            type_stats[f'{period}d_count'] = int(len(valid))

        stats[signal_type] = type_stats

    return stats


def get_current_signal_status(pcr_df, low_range=(0.6, 0.7), high_range=(1.5, 1.6)):
    """
    Determine the current market signal status based on latest PCR value.
    
    Returns:
        dict with: current_pcr, status, emoji, color, description
    """
    if pcr_df.empty:
        return {
            'current_pcr': None, 'status': 'No Data',
            'emoji': '❓', 'color': '#888',
            'description': 'No PCR data available'
        }

    latest = pcr_df.iloc[-1]
    pcr_val = latest['pcr']

    if pcr_val <= low_range[1]:
        if pcr_val <= low_range[0]:
            return {
                'current_pcr': pcr_val, 'status': 'EXTREME LOW',
                'emoji': '🟢', 'color': '#00ff88',
                'description': f'PCR at {pcr_val:.2f} — Strong buy signal zone (your strategy: expect rally in 1-2 weeks)'
            }
        return {
            'current_pcr': pcr_val, 'status': 'LOW — BUY ZONE',
            'emoji': '🟢', 'color': '#00ff88',
            'description': f'PCR at {pcr_val:.2f} — In your buy zone ({low_range[0]}-{low_range[1]}). Watch for rally.'
        }
    elif pcr_val >= high_range[0]:
        if pcr_val >= high_range[1]:
            return {
                'current_pcr': pcr_val, 'status': 'EXTREME HIGH',
                'emoji': '🔴', 'color': '#ff4757',
                'description': f'PCR at {pcr_val:.2f} — Strong caution zone (profit booking expected)'
            }
        return {
            'current_pcr': pcr_val, 'status': 'HIGH — CAUTION',
            'emoji': '🔴', 'color': '#ff4757',
            'description': f'PCR at {pcr_val:.2f} — In your caution zone ({high_range[0]}-{high_range[1]}). Watch for correction.'
        }
    elif pcr_val < 0.85:
        return {
            'current_pcr': pcr_val, 'status': 'BELOW NEUTRAL',
            'emoji': '🟡', 'color': '#ffaa00',
            'description': f'PCR at {pcr_val:.2f} — Below neutral, approaching buy zone'
        }
    elif pcr_val > 1.2:
        return {
            'current_pcr': pcr_val, 'status': 'ABOVE NEUTRAL',
            'emoji': '🟠', 'color': '#ff8c00',
            'description': f'PCR at {pcr_val:.2f} — Above neutral, approaching caution zone'
        }
    else:
        return {
            'current_pcr': pcr_val, 'status': 'NEUTRAL',
            'emoji': '⚪', 'color': '#00d4ff',
            'description': f'PCR at {pcr_val:.2f} — Neutral territory (0.85 - 1.20)'
        }
