"""
PCR Strategy Dashboard — Streamlit Application
================================================
Interactive dashboard for analyzing the Put-Call Ratio (PCR) contrarian strategy.
Features:
  - PCR time series with moving averages
  - Signal validation with vertical line pairs (trigger + 2 weeks later)
  - Nifty price overlay for correlation
  - Dynamic date range selection
  - Strategy performance statistics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_fetcher import fetch_pcr_history, fetch_nifty_prices, generate_synthetic_data
from pcr_calculator import (
    add_moving_averages, add_z_scores, add_bollinger_bands,
    find_signals, compute_forward_returns, compute_strategy_stats,
    get_current_signal_status,
)

def _dedent_html(html: str) -> str:
    """Strip leading whitespace from every line so Streamlit's markdown
    parser doesn't mistake indented HTML for a code block."""
    return "\n".join(line.lstrip() for line in html.split("\n"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="PCR Strategy Dashboard | Nifty 50",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOM CSS — Premium Dark Theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(160deg, #0a0a1a 0%, #0f1528 40%, #141e30 70%, #0a0a1a 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0 !important;
        font-family: 'Inter', sans-serif;
    }

    /* ── Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 212, 255, 0.15);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.05);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .metric-card.blue::before { background: linear-gradient(90deg, #00d4ff, #0099cc); }
    .metric-card.green::before { background: linear-gradient(90deg, #00ff88, #00cc6a); }
    .metric-card.amber::before { background: linear-gradient(90deg, #ffaa00, #ff8800); }
    .metric-card.red::before { background: linear-gradient(90deg, #ff4757, #ff2233); }
    .metric-card.purple::before { background: linear-gradient(90deg, #a855f7, #7c3aed); }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 12px;
        font-weight: 500;
        color: #9ca3af;
    }

    /* ── Title ── */
    .dashboard-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }
    .dashboard-subtitle {
        font-size: 13px;
        color: #6b7280;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* ── Signal Badge ── */
    .signal-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 24px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .signal-badge.bullish {
        background: rgba(0, 255, 136, 0.12);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.25);
    }
    .signal-badge.bearish {
        background: rgba(255, 71, 87, 0.12);
        color: #ff4757;
        border: 1px solid rgba(255, 71, 87, 0.25);
    }
    .signal-badge.neutral {
        background: rgba(0, 212, 255, 0.12);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.25);
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #e0e0e0;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Info boxes ── */
    .info-box {
        background: rgba(0, 212, 255, 0.06);
        border: 1px solid rgba(0, 212, 255, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        color: #b0c4de;
        font-size: 13px;
        line-height: 1.6;
    }

    /* ── Strategy explanation ── */
    .strategy-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .tag-green { background: rgba(0,255,136,0.12); color: #00ff88; }
    .tag-red { background: rgba(255,71,87,0.12); color: #ff4757; }

    /* ── Stats table ── */
    .stats-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 13px;
    }
    .stats-table th {
        background: rgba(255,255,255,0.04);
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 1px;
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .stats-table td {
        padding: 10px 14px;
        color: #d1d5db;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .stats-table tr:hover td {
        background: rgba(0, 212, 255, 0.03);
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* ── Hide Plotly modebar on mobile to prevent overlap with rangeselector ── */
    @media (max-width: 768px) {
        .modebar-container {
            display: none !important;
        }

    /* ── Streamlit overrides ── */
    .stSelectbox label, .stSlider label, .stDateInput label, .stRadio label {
        color: #9ca3af !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 600;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 255, 0.1) !important;
        color: #00d4ff !important;
    }
    
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_metric_card(label, value, delta=None, color_class="blue"):
    """Render a styled metric card."""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-card {color_class}">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {'#00d4ff' if color_class == 'blue' else '#00ff88' if color_class == 'green' else '#ffaa00' if color_class == 'amber' else '#ff4757' if color_class == 'red' else '#a855f7'};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def build_pcr_chart(pcr_df, nifty_df, signals_df, low_range, high_range,
                    show_ma5, show_ma21, show_nifty, show_signals, show_zones):
    """Build the main interactive PCR chart with all overlays."""

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("", ""),
    )

    # ── PCR Line ──
    fig.add_trace(go.Scatter(
        x=pcr_df['date'], y=pcr_df['pcr'],
        name='PCR (OI)',
        mode='lines',
        line=dict(color='#00d4ff', width=1.8),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.05)',
        hovertemplate='<b>Date</b>: %{x|%d-%b-%Y}<br><b>PCR</b>: %{y:.3f}<extra></extra>',
    ), row=1, col=1)

    # ── Moving Averages ──
    if show_ma5 and 'pcr_ma5' in pcr_df.columns:
        fig.add_trace(go.Scatter(
            x=pcr_df['date'], y=pcr_df['pcr_ma5'],
            name='MA-5',
            mode='lines',
            line=dict(color='#ffd700', width=1.5, dash='dot'),
            hovertemplate='<b>MA-5</b>: %{y:.3f}<extra></extra>',
        ), row=1, col=1)

    if show_ma21 and 'pcr_ma21' in pcr_df.columns:
        fig.add_trace(go.Scatter(
            x=pcr_df['date'], y=pcr_df['pcr_ma21'],
            name='MA-21',
            mode='lines',
            line=dict(color='#ff8c00', width=1.5, dash='dash'),
            hovertemplate='<b>MA-21</b>: %{y:.3f}<extra></extra>',
        ), row=1, col=1)

    # ── Threshold Zones ──
    if show_zones:
        # Low PCR zone (buy zone)
        fig.add_hrect(
            y0=low_range[0], y1=low_range[1],
            fillcolor="rgba(0, 255, 136, 0.06)",
            line=dict(width=0),
            annotation_text="BUY ZONE",
            annotation_position="top left",
            annotation_font=dict(size=10, color="rgba(0,255,136,0.5)"),
            row=1, col=1,
        )
        # High PCR zone (caution zone)
        fig.add_hrect(
            y0=high_range[0], y1=high_range[1],
            fillcolor="rgba(255, 71, 87, 0.06)",
            line=dict(width=0),
            annotation_text="CAUTION ZONE",
            annotation_position="top left",
            annotation_font=dict(size=10, color="rgba(255,71,87,0.5)"),
            row=1, col=1,
        )

    # ── Horizontal Reference Lines ──
    for val, color, dash_style in [
        (1.0, 'rgba(255,255,255,0.15)', 'dash'),
        (low_range[0], 'rgba(0,255,136,0.3)', 'dot'),
        (low_range[1], 'rgba(0,255,136,0.3)', 'dot'),
        (high_range[0], 'rgba(255,71,87,0.3)', 'dot'),
        (high_range[1], 'rgba(255,71,87,0.3)', 'dot'),
    ]:
        fig.add_hline(
            y=val, line_dash=dash_style,
            line_color=color, line_width=1,
            row=1, col=1,
        )

    # ── Signal Vertical Lines (Strategy Validator) ──
    if show_signals and not signals_df.empty:
        for _, sig in signals_df.iterrows():
            is_low = sig['signal_type'] == 'low_pcr'
            color = 'rgba(0, 255, 136, 0.5)' if is_low else 'rgba(255, 71, 87, 0.5)'
            fill = 'rgba(0, 255, 136, 0.04)' if is_low else 'rgba(255, 71, 87, 0.04)'
            label = 'BUY' if is_low else 'SELL'

            sig_date = pd.Timestamp(sig['signal_date'])
            tgt_date = pd.Timestamp(sig['target_date'])

            # Signal trigger line
            fig.add_vline(
                x=sig_date, line_dash='solid',
                line_color=color, line_width=1.5,
                row=1, col=1,
            )
            # Target date line (2 weeks later)
            fig.add_vline(
                x=tgt_date, line_dash='dash',
                line_color=color, line_width=1,
                row=1, col=1,
            )
            # Shaded region between signal and target
            fig.add_vrect(
                x0=sig_date, x1=tgt_date,
                fillcolor=fill, line_width=0,
                row=1, col=1,
            )

            # Also draw on Nifty subplot
            fig.add_vline(x=sig_date, line_dash='solid', line_color=color, line_width=1, row=2, col=1)
            fig.add_vline(x=tgt_date, line_dash='dash', line_color=color, line_width=1, row=2, col=1)
            fig.add_vrect(x0=sig_date, x1=tgt_date, fillcolor=fill, line_width=0, row=2, col=1)

    # ── Nifty Price (Bottom Subplot) ──
    if show_nifty and not nifty_df.empty:
        merged = pd.merge(pcr_df[['date']], nifty_df[['date', 'nifty_close']], on='date', how='inner')
        if not merged.empty:
            fig.add_trace(go.Scatter(
                x=merged['date'], y=merged['nifty_close'],
                name='Nifty 50',
                mode='lines',
                line=dict(color='#a855f7', width=1.8),
                fill='tozeroy',
                fillcolor='rgba(168, 85, 247, 0.05)',
                hovertemplate='<b>Nifty</b>: ₹%{y:,.0f}<extra></extra>',
            ), row=2, col=1)

    # ── Layout ──
    fig.update_layout(
        height=650,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.15)',
        font=dict(family='Inter, sans-serif', size=12, color='#9ca3af'),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='center', x=0.5,
            font=dict(size=11),
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='rgba(255,255,255,0.05)',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        xaxis2=dict(
            rangeslider=dict(visible=True, thickness=0.04),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(label="ALL", step="all"),
                ],
                bgcolor='rgba(255,255,255,0.05)',
                activecolor='rgba(0,212,255,0.2)',
                font=dict(size=11, color='#9ca3af'),
                bordercolor='rgba(255,255,255,0.08)',
                borderwidth=1,
                x=0, y=1.15,
            ),
        ),
    )

    # Y-axis labels
    fig.update_yaxes(
        title_text="PCR (Put OI / Call OI)", title_font=dict(size=12, color='#6b7280'),
        gridcolor='rgba(255,255,255,0.03)', zeroline=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Nifty 50", title_font=dict(size=12, color='#6b7280'),
        gridcolor='rgba(255,255,255,0.03)', zeroline=False,
        row=2, col=1,
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)', row=1, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)', row=2, col=1)

    return fig


def build_signal_table_html(signals_df):
    """Build styled HTML table of signals."""
    if signals_df.empty:
        return '<div class="info-box">No signals found in the selected date range with current thresholds.</div>'

    rows_html = ""
    for _, sig in signals_df.iterrows():
        is_low = sig['signal_type'] == 'low_pcr'
        badge = '<span class="signal-badge bullish">BUY</span>' if is_low else '<span class="signal-badge bearish">SELL</span>'
        sig_date = pd.Timestamp(sig['signal_date']).strftime('%d-%b-%Y')
        tgt_date = pd.Timestamp(sig['target_date']).strftime('%d-%b-%Y')
        pcr_val = f"{sig['pcr_value']:.3f}"

        nifty_val = f"{sig['nifty_at_signal']:,.0f}" if pd.notna(sig.get('nifty_at_signal')) else '—'

        ret_cells = ""
        for period in [5, 10, 15]:
            col = f'return_{period}d'
            if col in sig.index and pd.notna(sig[col]):
                val = sig[col]
                color = '#00ff88' if val > 0 else '#ff4757' if val < 0 else '#9ca3af'
                arrow = '▲' if val > 0 else '▼' if val < 0 else '─'
                ret_cells += f'<td style="color:{color}; font-family: JetBrains Mono, monospace; font-weight:600;">{arrow} {val:+.2f}%</td>'
            else:
                ret_cells += '<td style="color:#4a4a5a;">—</td>'

        rows_html += f"""
        <tr>
            <td>{badge}</td>
            <td style="font-family: JetBrains Mono, monospace;">{sig_date}</td>
            <td style="font-family: JetBrains Mono, monospace; color:#00d4ff;">{pcr_val}</td>
            <td style="font-family: JetBrains Mono, monospace;">{nifty_val}</td>
            <td style="font-family: JetBrains Mono, monospace;">{tgt_date}</td>
            {ret_cells}
        </tr>
        """

    html = f"""
    <div style="overflow-x:auto; border-radius:12px; border:1px solid rgba(255,255,255,0.05);">
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Signal</th>
                    <th>Date</th>
                    <th>PCR</th>
                    <th>Nifty</th>
                    <th>Target (+2w)</th>
                    <th>5-Day Return</th>
                    <th>10-Day Return</th>
                    <th>15-Day Return</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return _dedent_html(html)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown('<div class="dashboard-title" style="font-size:22px;">⚡ PCR Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Nifty 50 Put-Call Ratio Strategy</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Data Source ──
    st.markdown("### 📡 Data Source")
    data_source = st.radio(
        "Select data source",
        ["Sample Data (Demo)", "Live NSE (nselib)", "Cached Data"],
        index=2,
        help="Start with Sample Data to explore the dashboard. Switch to Live NSE to fetch real data."
    )

    st.markdown("---")

    # ── Date Range ──
    st.markdown("### 📅 Date Range")
    today = datetime.today().date()
    preset = st.selectbox(
        "Quick select",
        ["Last 1 Month", "Last 3 Months", "Last 6 Months", "Last 1 Year", "Custom"],
        index=2,
    )

    preset_days = {
        "Last 1 Month": 30,
        "Last 3 Months": 90,
        "Last 6 Months": 180,
        "Last 1 Year": 365,
    }

    if preset == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From", today - timedelta(days=180))
        with col2:
            end_date = st.date_input("To", today)
    else:
        days = preset_days[preset]
        start_date = today - timedelta(days=days)
        end_date = today

    st.markdown("---")

    # ── Strategy Thresholds ──
    st.markdown("### 🎯 PCR Thresholds")

    with st.expander("Buy Zone (Low PCR)", expanded=True):
        low_min = st.slider("Lower bound", 0.3, 1.0, 0.6, 0.05, key="low_min")
        low_max = st.slider("Upper bound", 0.3, 1.0, 0.7, 0.05, key="low_max")
        low_range = (min(low_min, low_max), max(low_min, low_max))

    with st.expander("Sell Zone (High PCR)", expanded=True):
        high_min = st.slider("Lower bound", 1.0, 2.5, 1.4, 0.02, key="high_min")
        high_max = st.slider("Upper bound", 1.0, 2.5, 1.5, 0.02, key="high_max")
        high_range = (min(high_min, high_max), max(high_min, high_max))

    st.markdown("---")

    # ── Display Options ──
    st.markdown("### 📊 Display Options")
    show_ma5 = st.checkbox("MA-5 (1 week)", value=False)
    show_ma21 = st.checkbox("MA-21 (1 month)", value=False)
    show_nifty = st.checkbox("Nifty Price Overlay", value=True)
    show_signals = st.checkbox("Signal Lines", value=True)
    show_zones = st.checkbox("Threshold Zones", value=True)
    use_smoothed = st.checkbox("Use smoothed PCR for signals", value=False,
                                help="Use MA-5 smoothed PCR instead of raw PCR for signal detection")

    st.markdown("---")
    st.markdown("""
    <div class="info-box" style="font-size:11px;">
        <strong>Strategy Logic</strong><br>
        <span class="strategy-tag tag-green">BUY</span> PCR in {:.2f}–{:.2f} → expect rally in 1-2 weeks<br>
        <span class="strategy-tag tag-red">SELL</span> PCR in {:.2f}–{:.2f} → expect profit booking
    </div>
    """.format(low_range[0], low_range[1], high_range[0], high_range[1]), unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_data(ttl=3600, show_spinner=False)
def load_synthetic(start, end):
    return generate_synthetic_data(start, end)
@st.cache_data(ttl=3600, show_spinner=False)
def load_nifty(start, end):
    return fetch_nifty_prices(start, end)


# ── Title ──
st.markdown("""
<div style="margin-bottom: 28px;">
    <div class="dashboard-title">📊 PCR Strategy Dashboard</div>
    <div class="dashboard-subtitle">
        Nifty 50 Put-Call Ratio Analysis &amp; Contrarian Signal Validation
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load Data ──
pcr_df = pd.DataFrame()
nifty_df = pd.DataFrame()
data_loaded = False

if data_source == "Sample Data (Demo)":
    pcr_df, nifty_df = load_synthetic(str(start_date), str(end_date))
    data_loaded = True

elif data_source == "Live NSE (nselib)":
    st.info("⏳ Fetching live data from NSE... This may take a few minutes for the first load (data is cached locally).")
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_cb(pct, msg):
        progress_bar.progress(min(pct, 1.0))
        status_text.text(msg)

    pcr_df = fetch_pcr_history(str(start_date), str(end_date), progress_callback=progress_cb)
    nifty_df = load_nifty(str(start_date), str(end_date))

    progress_bar.progress(1.0)
    status_text.text("✅ Data loaded successfully!")

    if pcr_df.empty:
        st.warning("⚠️ No PCR data was fetched. NSE may be rate-limiting requests or nselib may not be installed. Try 'Sample Data (Demo)' mode.")
    else:
        data_loaded = True

elif data_source == "Cached Data":
    pcr_df = fetch_pcr_history(str(start_date), str(end_date))
    nifty_df = load_nifty(str(start_date), str(end_date))

    if pcr_df.empty:
        st.warning("⚠️ No cached data found. Fetch live data first, or try 'Sample Data (Demo)' mode.")
    else:
        data_loaded = True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN DASHBOARD (only if data is loaded)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if data_loaded and not pcr_df.empty:

    # ── Enrich PCR data ──
    pcr_df = add_moving_averages(pcr_df, periods=[5, 21])
    pcr_df = add_z_scores(pcr_df)
    pcr_df = add_bollinger_bands(pcr_df)

    # ── Find Signals ──
    signals_df = find_signals(
        pcr_df, low_range=low_range, high_range=high_range,
        use_smoothed=use_smoothed, ma_period=5,
    )

    # ── Compute Forward Returns ──
    if not signals_df.empty and not nifty_df.empty:
        signals_df = compute_forward_returns(pcr_df, nifty_df, signals_df, periods=[5, 10, 15])
        strategy_stats = compute_strategy_stats(signals_df, periods=[5, 10, 15])
    else:
        strategy_stats = {}

    # ── Signal Status ──
    status = get_current_signal_status(pcr_df, low_range, high_range)

    # ── KPI CARDS ──
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        pcr_val = status['current_pcr']
        render_metric_card(
            "Current PCR",
            f"{pcr_val:.3f}" if pcr_val else "—",
            f"Latest data point",
            "blue"
        )

    with col2:
        badge_class = "bullish" if "LOW" in status['status'] or "BELOW" in status['status'] else "bearish" if "HIGH" in status['status'] or "ABOVE" in status['status'] else "neutral"
        render_metric_card(
            "Signal Status",
            f"{status['emoji']} {status['status']}",
            color_class="green" if badge_class == "bullish" else "red" if badge_class == "bearish" else "blue"
        )

    with col3:
        low_count = len(signals_df[signals_df['signal_type'] == 'low_pcr']) if not signals_df.empty else 0
        render_metric_card("Buy Signals", str(low_count), f"PCR in {low_range[0]:.1f}–{low_range[1]:.1f}", "green")

    with col4:
        high_count = len(signals_df[signals_df['signal_type'] == 'high_pcr']) if not signals_df.empty else 0
        render_metric_card("Sell Signals", str(high_count), f"PCR in {high_range[0]:.1f}–{high_range[1]:.1f}", "red")

    with col5:
        total_days = len(pcr_df)
        render_metric_card("Trading Days", str(total_days), f"{start_date} → {end_date}", "purple")

    st.markdown("")

    # ── MAIN CHART ──
    st.markdown('<div class="section-header">📈 PCR Time Series with Signal Validation</div>', unsafe_allow_html=True)

    fig = build_pcr_chart(
        pcr_df, nifty_df, signals_df,
        low_range, high_range,
        show_ma5, show_ma21, show_nifty, show_signals, show_zones,
    )
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    })

    # ── Signal description ──
    st.markdown(f"""
    <div class="info-box">
        <strong>How to read the chart:</strong>
        Solid vertical lines mark when PCR entered a trigger zone.
        Dashed vertical lines mark 10 trading days (~2 weeks) later.
        The shaded region between them is your observation window.
        <span class="strategy-tag tag-green">GREEN</span> = Buy zone ({low_range[0]:.2f}–{low_range[1]:.2f})
        &nbsp;&nbsp;
        <span class="strategy-tag tag-red">RED</span> = Sell zone ({high_range[0]:.2f}–{high_range[1]:.2f})
        <br>Check if Nifty moved UP after green signals and DOWN after red signals to validate your strategy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TABS: Signal Table + Strategy Stats
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    tab1, tab2, tab3 = st.tabs(["📋 Signal Log", "📊 Strategy Performance", "ℹ️ About PCR"])

    # ── Tab 1: Signal Log ──
    with tab1:
        st.markdown('<div class="section-header">📋 All Detected Signals</div>', unsafe_allow_html=True)
        st.markdown(build_signal_table_html(signals_df), unsafe_allow_html=True)

    # ── Tab 2: Strategy Stats ──
    with tab2:
        st.markdown('<div class="section-header">📊 Strategy Performance Summary</div>', unsafe_allow_html=True)

        if strategy_stats:
            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                st.markdown("""
                <div class="glass-card">
                    <div style="font-size:15px; font-weight:700; color:#00ff88; margin-bottom:16px;">
                        🟢 BUY Signal Performance
                    </div>
                """, unsafe_allow_html=True)

                if 'low_pcr' in strategy_stats and strategy_stats['low_pcr']['total_signals'] > 0:
                    stats = strategy_stats['low_pcr']
                    rows = ""
                    for period in [5, 10, 15]:
                        avg = stats.get(f'{period}d_avg_return', '—')
                        wr = stats.get(f'{period}d_win_rate', '—')
                        mx = stats.get(f'{period}d_max_gain', '—')
                        mn = stats.get(f'{period}d_max_loss', '—')
                        avg_str = f"{avg:+.2f}%" if isinstance(avg, (int, float)) else avg
                        wr_str = f"{wr:.1f}%" if isinstance(wr, (int, float)) else wr
                        mx_str = f"{mx:+.2f}%" if isinstance(mx, (int, float)) else mx
                        mn_str = f"{mn:+.2f}%" if isinstance(mn, (int, float)) else mn

                        avg_color = '#00ff88' if isinstance(avg, (int, float)) and avg > 0 else '#ff4757'
                        wr_color = '#00ff88' if isinstance(wr, (int, float)) and wr >= 50 else '#ff4757'

                        rows += f"""
                        <tr>
                            <td style="font-weight:600;">{period}-Day</td>
                            <td style="color:{avg_color}; font-family:JetBrains Mono,monospace; font-weight:600;">{avg_str}</td>
                            <td style="color:{wr_color}; font-family:JetBrains Mono,monospace; font-weight:600;">{wr_str}</td>
                            <td style="color:#00ff88; font-family:JetBrains Mono,monospace;">{mx_str}</td>
                            <td style="color:#ff4757; font-family:JetBrains Mono,monospace;">{mn_str}</td>
                        </tr>
                        """

                    st.markdown(_dedent_html(f"""
                        <div style="font-size:12px; color:#9ca3af; margin-bottom:8px;">
                            Total signals: <strong style="color:#e0e0e0;">{stats['total_signals']}</strong>
                        </div>
                        <table class="stats-table">
                            <thead>
                                <tr><th>Period</th><th>Avg Return</th><th>Win Rate</th><th>Best</th><th>Worst</th></tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    """), unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#6b7280;">No buy signals found.</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with stat_col2:
                st.markdown("""
                <div class="glass-card">
                    <div style="font-size:15px; font-weight:700; color:#ff4757; margin-bottom:16px;">
                        🔴 SELL Signal Performance
                    </div>
                """, unsafe_allow_html=True)

                if 'high_pcr' in strategy_stats and strategy_stats['high_pcr']['total_signals'] > 0:
                    stats = strategy_stats['high_pcr']
                    rows = ""
                    for period in [5, 10, 15]:
                        avg = stats.get(f'{period}d_avg_return', '—')
                        wr = stats.get(f'{period}d_win_rate', '—')
                        mx = stats.get(f'{period}d_max_gain', '—')
                        mn = stats.get(f'{period}d_max_loss', '—')
                        avg_str = f"{avg:+.2f}%" if isinstance(avg, (int, float)) else avg
                        wr_str = f"{wr:.1f}%" if isinstance(wr, (int, float)) else wr
                        mx_str = f"{mx:+.2f}%" if isinstance(mx, (int, float)) else mx
                        mn_str = f"{mn:+.2f}%" if isinstance(mn, (int, float)) else mn

                        avg_color = '#ff4757' if isinstance(avg, (int, float)) and avg < 0 else '#00ff88'
                        wr_color = '#00ff88' if isinstance(wr, (int, float)) and wr >= 50 else '#ff4757'

                        rows += f"""
                        <tr>
                            <td style="font-weight:600;">{period}-Day</td>
                            <td style="color:{avg_color}; font-family:JetBrains Mono,monospace; font-weight:600;">{avg_str}</td>
                            <td style="color:{wr_color}; font-family:JetBrains Mono,monospace; font-weight:600;">{wr_str}</td>
                            <td style="color:#00ff88; font-family:JetBrains Mono,monospace;">{mx_str}</td>
                            <td style="color:#ff4757; font-family:JetBrains Mono,monospace;">{mn_str}</td>
                        </tr>
                        """

                    st.markdown(f"""
                        <div style="font-size:12px; color:#9ca3af; margin-bottom:8px;">
                            Total signals: <strong style="color:#e0e0e0;">{stats['total_signals']}</strong>
                        </div>
                        <table class="stats-table">
                            <thead>
                                <tr><th>Period</th><th>Avg Return</th><th>Win Rate</th><th>Best</th><th>Worst</th></tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#6b7280;">No sell signals found.</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # ── Interpretation ──
            st.markdown("")
            st.markdown("""
            <div class="info-box">
                <strong>How to interpret:</strong><br>
                • <strong>Win Rate &gt; 60%</strong>: Your strategy has a meaningful edge at this threshold<br>
                • <strong>Win Rate 50-60%</strong>: Slight edge, may work with proper risk management<br>
                • <strong>Win Rate &lt; 50%</strong>: Strategy is not effective at these thresholds — try adjusting them<br>
                • <strong>Avg Return</strong>: Positive for BUY signals and negative for SELL signals = strategy is working<br>
                <br>
                💡 <em>Tip: Adjust the PCR thresholds in the sidebar and observe how win rates change. This is your edge!</em>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">No signals detected. Try widening the PCR threshold ranges in the sidebar.</div>', unsafe_allow_html=True)

    # ── Tab 3: About PCR ──
    with tab3:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header" style="margin-top:0;">What is the Put-Call Ratio (PCR)?</div>
            <div style="color:#b0c4de; line-height:1.8; font-size:14px;">
                <p>The <strong>Put-Call Ratio (PCR)</strong> is a sentiment indicator calculated as:</p>
                <div style="background:rgba(0,212,255,0.08); padding:12px 20px; border-radius:8px; font-family:JetBrains Mono,monospace; font-size:16px; text-align:center; margin:16px 0; color:#00d4ff;">
                    PCR = Total Put Open Interest ÷ Total Call Open Interest
                </div>
                <p><strong>Contrarian Interpretation:</strong></p>
                <ul>
                    <li><span class="strategy-tag tag-green">PCR &lt; 0.7</span>
                        Too many calls (bullish bets) → market is complacent → <strong>contrarian buy</strong> or <strong>potential reversal</strong></li>
                    <li><span class="strategy-tag tag-red">PCR &gt; 1.5</span>
                        Too many puts (bearish bets) → market is fearful → <strong>contrarian sell/profit-booking</strong></li>
                    <li><strong>PCR ~ 1.0</strong> → Neutral sentiment, no extreme positioning</li>
                </ul>
                <p><strong>Indian Market Nuance:</strong> In Nifty options, put <em>writing</em> (selling puts) is a common bullish strategy.
                So high PCR can sometimes indicate strong support, not fear. This is why visual validation with this dashboard is crucial!</p>
                <p><strong>Best Practices:</strong></p>
                <ul>
                    <li>Use <strong>OI-based PCR</strong> for medium-term signals (1-2 weeks) — this is what we compute</li>
                    <li>Apply <strong>moving averages</strong> (MA-5, MA-21) to filter noise</li>
                    <li>Combine with <strong>Nifty price action</strong> and <strong>VIX</strong> for confirmation</li>
                    <li>Use <strong>statistical extremes</strong> (Z-scores) rather than fixed thresholds</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif not data_loaded:
    # ── Empty State ──
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <div style="font-size:64px; margin-bottom:20px;">📊</div>
        <div style="font-size:22px; font-weight:700; color:#e0e0e0; margin-bottom:12px;">
            Select a Data Source to Begin
        </div>
        <div style="font-size:14px; color:#6b7280; max-width:500px; margin:0 auto; line-height:1.8;">
            Choose <strong style="color:#00d4ff;">"Sample Data (Demo)"</strong> in the sidebar to explore the dashboard immediately,
            or <strong style="color:#00ff88;">"Live NSE (nselib)"</strong> to fetch real Nifty PCR data from NSE.
        </div> 
    </div>
    """, unsafe_allow_html=True)
