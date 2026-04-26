"""
RELIANCE Industries — AttentionGRU Forecast Dashboard
Streamlit deployment for AttentionGRU_v2 / v4 model
Run: streamlit run app.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RELIANCE · GRU Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
DARK_BG      = "#0B0F1A"
CARD_BG      = "#111827"
BORDER       = "#1E293B"
ACCENT_BLUE  = "#3B82F6"
ACCENT_TEAL  = "#14B8A6"
ACCENT_AMBER = "#F59E0B"
ACCENT_RED   = "#EF4444"
ACCENT_GREEN = "#22C55E"
TEXT_PRIMARY = "#F1F5F9"
TEXT_MUTED   = "#64748B"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] {{
      background-color: {DARK_BG};
      color: {TEXT_PRIMARY};
      font-family: 'Inter', sans-serif;
  }}

  /* Header */
  .hero-header {{
      background: linear-gradient(135deg, #0B0F1A 0%, #0F172A 50%, #0B1629 100%);
      border-bottom: 1px solid {BORDER};
      padding: 2.5rem 2rem 2rem;
      margin: -1rem -1rem 2rem -1rem;
      position: relative;
      overflow: hidden;
  }}
  .hero-header::before {{
      content: '';
      position: absolute;
      top: -60px; right: -60px;
      width: 300px; height: 300px;
      background: radial-gradient(circle, {ACCENT_BLUE}18 0%, transparent 70%);
      pointer-events: none;
  }}
  .hero-title {{
      font-family: 'Syne', sans-serif;
      font-size: 2.2rem;
      font-weight: 800;
      color: {TEXT_PRIMARY};
      letter-spacing: -0.02em;
      margin: 0 0 0.25rem;
  }}
  .hero-subtitle {{
      font-family: 'DM Mono', monospace;
      font-size: 0.78rem;
      color: {ACCENT_TEAL};
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin: 0;
  }}
  .hero-badge {{
      display: inline-block;
      background: {ACCENT_BLUE}22;
      border: 1px solid {ACCENT_BLUE}55;
      color: {ACCENT_BLUE};
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      padding: 0.2rem 0.6rem;
      border-radius: 4px;
      margin-left: 1rem;
      vertical-align: middle;
  }}

  /* Cards */
  .metric-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      position: relative;
      overflow: hidden;
  }}
  .metric-card::after {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg, {ACCENT_BLUE}, {ACCENT_TEAL});
  }}
  .metric-label {{
      font-family: 'DM Mono', monospace;
      font-size: 0.68rem;
      color: {TEXT_MUTED};
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.4rem;
  }}
  .metric-value {{
      font-family: 'Syne', sans-serif;
      font-size: 1.8rem;
      font-weight: 700;
      color: {TEXT_PRIMARY};
      line-height: 1;
  }}
  .metric-sub {{
      font-size: 0.75rem;
      color: {TEXT_MUTED};
      margin-top: 0.3rem;
  }}
  .metric-good  {{ color: {ACCENT_GREEN} !important; }}
  .metric-warn  {{ color: {ACCENT_AMBER} !important; }}
  .metric-bad   {{ color: {ACCENT_RED}   !important; }}

  /* Section headers */
  .section-title {{
      font-family: 'Syne', sans-serif;
      font-size: 1.1rem;
      font-weight: 700;
      color: {TEXT_PRIMARY};
      letter-spacing: -0.01em;
      border-left: 3px solid {ACCENT_BLUE};
      padding-left: 0.75rem;
      margin: 2rem 0 1rem;
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background-color: #0D1321;
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] .sidebar-label {{
      font-family: 'DM Mono', monospace;
      font-size: 0.7rem;
      color: {TEXT_MUTED};
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 0.5rem;
  }}

  /* Forecast table */
  .forecast-row {{
      display: flex;
      align-items: center;
      padding: 0.55rem 1rem;
      border-bottom: 1px solid {BORDER};
      font-size: 0.85rem;
  }}
  .forecast-row:last-child {{ border-bottom: none; }}
  .forecast-row:hover {{ background: {BORDER}44; }}
  .forecast-date {{ font-family: 'DM Mono', monospace; color: {TEXT_MUTED}; flex: 1; }}
  .forecast-price {{ font-family: 'DM Mono', monospace; font-weight: 500; flex: 1; text-align: right; }}
  .forecast-pct {{ flex: 1; text-align: right; font-family: 'DM Mono', monospace; font-size: 0.8rem; }}
  .up   {{ color: {ACCENT_GREEN}; }}
  .down {{ color: {ACCENT_RED};   }}

  /* Architecture pills */
  .arch-grid {{
      display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem;
  }}
  .arch-pill {{
      background: {BORDER};
      border: 1px solid #2D3748;
      border-radius: 6px;
      padding: 0.3rem 0.7rem;
      font-family: 'DM Mono', monospace;
      font-size: 0.72rem;
      color: {TEXT_MUTED};
  }}
  .arch-pill span {{ color: {ACCENT_TEAL}; font-weight: 500; }}

  /* Disclaimer */
  .disclaimer {{
      background: {ACCENT_AMBER}11;
      border: 1px solid {ACCENT_AMBER}44;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      font-size: 0.78rem;
      color: {ACCENT_AMBER};
      margin-top: 1.5rem;
  }}

  /* Plotly overrides */
  .js-plotly-plot .plotly {{ background: transparent !important; }}

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 0.5rem;
      border-bottom: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
      font-family: 'DM Mono', monospace;
      font-size: 0.78rem;
      letter-spacing: 0.05em;
      color: {TEXT_MUTED};
      border-radius: 6px 6px 0 0;
      padding: 0.5rem 1rem;
  }}
  .stTabs [aria-selected="true"] {{
      color: {TEXT_PRIMARY} !important;
      background: {CARD_BG} !important;
      border-top: 2px solid {ACCENT_BLUE} !important;
  }}

  /* Hide Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding-top: 0 !important; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: Plotly dark theme ─────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0D1321",
    font=dict(family="DM Mono, monospace", color=TEXT_MUTED, size=11),
    xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False,
               tickfont=dict(size=10), linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False,
               tickfont=dict(size=10), linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
                font=dict(size=10)),
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
)


# ── Model artifact loader ─────────────────────────────────────────────────────
MODEL_DIR = "outputs/models"

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts(model_dir: str):
    try:
        import tensorflow as tf

        class BahdanauAttention(tf.keras.layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.units = units
                self.W = tf.keras.layers.Dense(units)
                self.V = tf.keras.layers.Dense(1)
            def call(self, h):
                s = self.V(tf.nn.tanh(self.W(h)))
                s = s - tf.reduce_max(s, axis=1, keepdims=True)
                a = tf.nn.softmax(s, axis=1)
                return tf.reduce_sum(a * h, axis=1), a
            def get_config(self):
                cfg = super().get_config()
                cfg.update({"units": self.units})
                return cfg

        with open(os.path.join(model_dir, "model_config.json")) as f:
            config = json.load(f)
        model    = tf.keras.models.load_model(
            os.path.join(model_dir, "final_model.keras"),
            custom_objects={"BahdanauAttention": BahdanauAttention})
        f_scaler = joblib.load(os.path.join(model_dir, "final_f_scaler.pkl"))
        t_scaler = joblib.load(os.path.join(model_dir, "final_t_scaler.pkl"))
        return model, f_scaler, t_scaler, config, None
    except Exception as e:
        return None, None, None, None, str(e)


# ── Feature engineering (mirrors training) ────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"]      = df["Close"].pct_change()
    df["Log_Return"]  = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Ratio"]    = (df["High"] - df["Low"]) / df["Close"]
    df["OC_Ratio"]    = (df["Close"] - df["Open"]) / df["Open"]
    df["Momentum_5"]  = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    df["Vol_10"]      = df["Return"].rolling(10).std()
    df["MA5"]         = df["Close"].rolling(5).mean()
    df["MA20"]        = df["Close"].rolling(20).mean()
    df["MA_Ratio"]    = df["MA5"] / df["MA20"]
    df.drop(columns=["MA5", "MA20"], inplace=True)
    df["Log_Volume"]  = np.log(df["Volume"].clip(lower=1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ── Forecast engine ───────────────────────────────────────────────────────────
def forecast_n_days(model, f_scaler, t_scaler, config, df_feat, n_days):
    seq_len      = config["sequence_len"]
    feature_cols = config["feature_cols"]
    if len(df_feat) < seq_len:
        return None, f"Need at least {seq_len} rows of data, got {len(df_feat)}"

    last_rows      = df_feat[feature_cols].iloc[-seq_len:].values
    seq            = f_scaler.transform(last_rows)[np.newaxis, :, :]
    avg_log_volume = float(df_feat["Log_Volume"].iloc[-30:].mean())
    avg_hl_ratio   = float(df_feat["HL_Ratio"].iloc[-30:].mean())
    last_close     = float(df_feat["Close"].iloc[-1])
    last_date      = df_feat.index[-1]

    close_idx    = feature_cols.index("Close")
    raw_close_col = f_scaler.transform(last_rows)[:, close_idx]
    close_buffer = list(last_rows[:, close_idx])
    lr_buffer    = []
    results      = []

    for day in range(n_days):
        pred_s  = model.predict(seq, verbose=0)
        pred_lr = float(t_scaler.inverse_transform(pred_s)[0, 0])
        prev_c  = close_buffer[-1]
        new_c   = prev_c * np.exp(pred_lr)
        close_buffer.append(new_c)
        lr_buffer.append(pred_lr)
        results.append({
            "day"      : day + 1,
            "pred_lr"  : pred_lr,
            "new_close": new_c,
            "pct"      : (new_c / last_close - 1) * 100,
            "direction": "UP" if pred_lr > 0 else "DOWN",
        })
        cb = close_buffer
        raw_row = np.array([[
            prev_c,
            new_c * (1 + avg_hl_ratio / 2),
            new_c * (1 - avg_hl_ratio / 2),
            new_c, np.exp(avg_log_volume), pred_lr, avg_hl_ratio,
            (new_c - prev_c) / prev_c,
            (new_c - cb[-6])  if len(cb) > 5  else 0.0,
            (new_c - cb[-11]) if len(cb) > 10 else 0.0,
            float(np.std(lr_buffer[-9:] + [pred_lr])) if lr_buffer else 0.0,
            (np.mean(cb[-5:]) / np.mean(cb[-20:])) if len(cb) >= 20 else 1.0,
            avg_log_volume,
        ]])
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, :] = f_scaler.transform(raw_row)[0]

    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    out = pd.DataFrame(results)
    out["Date"] = future_dates
    return out, None


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                color:#F1F5F9;padding:0.5rem 0 1.5rem;border-bottom:1px solid #1E293B;
                margin-bottom:1.5rem;'>
        ⚙️ Configuration
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>Model artifacts directory</div>", unsafe_allow_html=True)
    model_dir = st.text_input("", value="outputs/models", label_visibility="collapsed")

    st.markdown("<div class='sidebar-label' style='margin-top:1rem'>Stock data CSV</div>", unsafe_allow_html=True)
    csv_file = st.file_uploader("", type=["csv"], label_visibility="collapsed",
                                 help="Upload OHLCV CSV with columns: Date, Open, High, Low, Close, Volume")

    st.markdown("<div class='sidebar-label' style='margin-top:1rem'>Forecast horizon</div>", unsafe_allow_html=True)
    n_days = st.slider("", min_value=1, max_value=30, value=10,
                        label_visibility="collapsed",
                        help="Number of business days to forecast ahead")

    st.markdown("<div class='sidebar-label' style='margin-top:1rem'>Historical chart window</div>", unsafe_allow_html=True)
    hist_days = st.select_slider("", options=[30, 60, 90, 180, 365], value=90,
                                  label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1E293B;margin:1.5rem 0'>", unsafe_allow_html=True)

    # Pre-loaded forecast CSV (optional)
    st.markdown("<div class='sidebar-label'>Load saved forecast CSV</div>", unsafe_allow_html=True)
    forecast_csv = st.file_uploader("", type=["csv"], key="forecast_csv",
                                     label_visibility="collapsed",
                                     help="Optionally load a pre-generated forecast CSV")

    st.markdown("<div class='sidebar-label' style='margin-top:1rem'>Load saved metrics CSV</div>", unsafe_allow_html=True)
    metrics_csv = st.file_uploader("", type=["csv"], key="metrics_csv",
                                    label_visibility="collapsed")

    st.markdown("""
    <div style='margin-top:2rem;font-family:DM Mono,monospace;font-size:0.65rem;
                color:#334155;line-height:1.6;'>
        AttentionGRU_v4 · RELIANCE Industries<br>
        Walk-forward cross-validation<br>
        Huber loss · Adam optimizer
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-header'>
  <p class='hero-subtitle'>NSE · RELIANCE · Attention-GRU Forecasting System</p>
  <h1 class='hero-title'>RELIANCE Stock Forecast
    <span class='hero-badge'>AttentionGRU_v4</span>
  </h1>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  LOAD ARTIFACTS
# ════════════════════════════════════════════════════════════════════════════════
model, f_scaler, t_scaler, config, load_err = load_artifacts(model_dir)

if load_err:
    st.error(f"**Model loading failed:** {load_err}")
    st.info("""
    **Expected directory structure:**
    ```
    outputs/models/
    ├── final_model.keras
    ├── final_f_scaler.pkl
    ├── final_t_scaler.pkl
    └── model_config.json
    ```
    Update the path in the sidebar or place artifacts in `outputs/models/`.
    """)
    st.stop()

MODEL_LOADED = model is not None


# ════════════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════
df_raw = None
df_feat = None
data_err = None

if csv_file is not None:
    try:
        df_raw = pd.read_csv(csv_file)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=True)
        df_raw.set_index("Date", inplace=True)
        df_raw.sort_index(inplace=True)
        df_raw = df_raw[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce")
        df_raw["Volume"] = df_raw["Volume"].replace(0, np.nan)
        df_raw.dropna(inplace=True)
        df_feat = add_features(df_raw)
    except Exception as e:
        data_err = str(e)


# ════════════════════════════════════════════════════════════════════════════════
#  TOP METRIC STRIP
# ════════════════════════════════════════════════════════════════════════════════
if config:
    arch  = config.get("architecture", {})
    train = config.get("training", {})

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Best Fold</div>
          <div class='metric-value'>#{config.get('trained_on_fold','—')}</div>
          <div class='metric-sub'>Selected by min val_loss</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Architecture</div>
          <div class='metric-value' style='font-size:1.1rem;padding-top:0.3rem'>
            GRU({arch.get('gru1','?')},{arch.get('gru2','?')})
          </div>
          <div class='metric-sub'>+ Bahdanau Attention</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Train end date</div>
          <div class='metric-value' style='font-size:1.1rem;padding-top:0.3rem'>
            {config.get('train_end_date','—')}
          </div>
          <div class='metric-sub'>Walk-forward fold cutoff</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Sequence length</div>
          <div class='metric-value'>{config.get('sequence_len','—')}<span style='font-size:1rem;color:{TEXT_MUTED}'> days</span></div>
          <div class='metric-sub'>{config.get('n_features','—')} input features</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Loss / Optimiser</div>
          <div class='metric-value' style='font-size:1.1rem;padding-top:0.3rem'>
            {train.get('loss','?').title()} / Adam
          </div>
          <div class='metric-sub'>LR {train.get('peak_lr','?')} peak · {train.get('batch_size','?')} batch</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════════════
tab_forecast, tab_metrics, tab_history, tab_arch = st.tabs([
    "📈  Live Forecast",
    "📊  Model Metrics",
    "📜  Price History",
    "🔬  Architecture",
])


# ─────────────────────────────────────────────────────────────────
#  TAB 1: LIVE FORECAST
# ─────────────────────────────────────────────────────────────────
with tab_forecast:

    # ── Try to produce a forecast ────────────────────────────────
    forecast_df = None
    fc_err = None

    if MODEL_LOADED and df_feat is not None:
        with st.spinner("Running inference…"):
            forecast_df, fc_err = forecast_n_days(
                model, f_scaler, t_scaler, config, df_feat, n_days)

    elif forecast_csv is not None:
        # Fallback: use pre-loaded CSV
        try:
            forecast_df = pd.read_csv(forecast_csv, parse_dates=["Date"])
            forecast_df["pct"] = forecast_df.get("Pct_vs_last", forecast_df.get("pct", 0))
            forecast_df["new_close"] = forecast_df.get("Predicted_Close", forecast_df.get("new_close", 0))
            forecast_df["direction"] = forecast_df["pct"].apply(lambda x: "UP" if x >= 0 else "DOWN")
            st.info("Showing pre-loaded forecast CSV (upload a stock CSV for live inference)")
        except Exception as e:
            fc_err = str(e)

    if fc_err:
        st.error(f"Forecast error: {fc_err}")

    if forecast_df is not None and not forecast_df.empty:
        last_close = float(df_feat["Close"].iloc[-1]) if df_feat is not None else None
        final_pred = float(forecast_df["new_close"].iloc[-1])
        final_pct  = float(forecast_df["pct"].iloc[-1])
        days_up    = int((forecast_df["direction"] == "UP").sum())
        days_down  = n_days - days_up

        # Summary metrics row
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            if last_close:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>Last Close</div>
                  <div class='metric-value'>₹{last_close:,.2f}</div>
                  <div class='metric-sub'>{df_feat.index[-1].date() if df_feat is not None else '—'}</div>
                </div>""", unsafe_allow_html=True)
        with mc2:
            color_cls = "metric-good" if final_pct >= 0 else "metric-bad"
            arrow = "▲" if final_pct >= 0 else "▼"
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Day {n_days} Target</div>
              <div class='metric-value class="{color_cls}"'>₹{final_pred:,.2f}</div>
              <div class='metric-sub {color_cls}'>{arrow} {abs(final_pct):.2f}% vs today</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Bullish days</div>
              <div class='metric-value metric-good'>{days_up}</div>
              <div class='metric-sub'>out of {n_days} forecast days</div>
            </div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Bearish days</div>
              <div class='metric-value metric-bad'>{days_down}</div>
              <div class='metric-sub'>out of {n_days} forecast days</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Chart ─────────────────────────────────────────────────
        fc1, fc2 = st.columns([2, 1])

        with fc1:
            st.markdown("<div class='section-title'>Price Forecast Trajectory</div>", unsafe_allow_html=True)
            fig = go.Figure()

            # Historical tail
            if df_raw is not None:
                hist_tail = df_raw["Close"].iloc[-hist_days:]
                fig.add_trace(go.Scatter(
                    x=hist_tail.index, y=hist_tail.values,
                    mode="lines", name="Historical",
                    line=dict(color=ACCENT_BLUE, width=1.5),
                    opacity=0.9,
                ))
                # Anchor point connector
                anchor_x = [hist_tail.index[-1], forecast_df["Date"].iloc[0]]
                anchor_y = [float(hist_tail.iloc[-1]), float(forecast_df["new_close"].iloc[0])]
                fig.add_trace(go.Scatter(
                    x=anchor_x, y=anchor_y,
                    mode="lines", name="",
                    line=dict(color=ACCENT_TEAL, width=1, dash="dot"),
                    showlegend=False,
                ))

            # Forecast band (±1% confidence shade)
            preds = forecast_df["new_close"].values
            upper = preds * 1.010
            lower = preds * 0.990
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=f"{ACCENT_TEAL}18",
                line=dict(color="rgba(0,0,0,0)"),
                name="±1% Band",
                hoverinfo="skip",
            ))
            # Forecast line
            colors = [ACCENT_GREEN if d == "UP" else ACCENT_RED
                      for d in forecast_df["direction"]]
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"], y=forecast_df["new_close"],
                mode="lines+markers", name="Forecast",
                line=dict(color=ACCENT_TEAL, width=2),
                marker=dict(color=colors, size=7, line=dict(color=DARK_BG, width=1)),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>",
            ))
            # Vertical "today" line
            if df_raw is not None:
                fig.add_vline(x=df_raw.index[-1], line_dash="dash",
                              line_color=TEXT_MUTED, opacity=0.5, annotation_text="Today")

            fig.update_layout(**PLOT_LAYOUT, height=360,
                              title=dict(text="", x=0))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with fc2:
            st.markdown("<div class='section-title'>Day-by-Day Table</div>", unsafe_allow_html=True)
            st.markdown("<div style='background:#111827;border:1px solid #1E293B;border-radius:10px;overflow:hidden;'>", unsafe_allow_html=True)
            # Header
            st.markdown(f"""
            <div class='forecast-row' style='border-bottom:1px solid #1E293B;opacity:0.5;font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;'>
              <span class='forecast-date'>Date</span>
              <span class='forecast-price'>Price ₹</span>
              <span class='forecast-pct'>Chg%</span>
            </div>""", unsafe_allow_html=True)

            for _, row in forecast_df.iterrows():
                d = row["Date"].strftime("%d %b") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
                p = f"{row['new_close']:,.2f}"
                pct_val = row["pct"]
                pct_str = f"+{pct_val:.2f}%" if pct_val >= 0 else f"{pct_val:.2f}%"
                dir_cls = "up" if row["direction"] == "UP" else "down"
                arrow = "▲" if row["direction"] == "UP" else "▼"
                st.markdown(f"""
                <div class='forecast-row'>
                  <span class='forecast-date'>{d}</span>
                  <span class='forecast-price'>{p}</span>
                  <span class='forecast-pct {dir_cls}'>{arrow} {pct_str}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Daily return bar chart
        st.markdown("<div class='section-title'>Predicted Daily Log Returns</div>", unsafe_allow_html=True)
        if "pred_lr" in forecast_df.columns:
            bar_colors = [ACCENT_GREEN if v >= 0 else ACCENT_RED for v in forecast_df["pred_lr"]]
            fig2 = go.Figure(go.Bar(
                x=forecast_df["Date"],
                y=forecast_df["pred_lr"] * 100,
                marker_color=bar_colors,
                hovertemplate="<b>%{x|%d %b}</b><br>%{y:.4f}%<extra></extra>",
            ))
            fig2.add_hline(y=0, line_color=TEXT_MUTED, line_dash="dash", opacity=0.5)
            fig2.update_layout(**PLOT_LAYOUT, height=200,
                               yaxis_title="Log return (%)")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    else:
        # Placeholder when no data yet
        st.markdown("""
        <div style='background:#111827;border:1px solid #1E293B;border-radius:12px;
                    padding:3rem;text-align:center;margin:1rem 0;'>
          <div style='font-size:2.5rem;margin-bottom:1rem'>📂</div>
          <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#94A3B8;
                      margin-bottom:0.5rem;'>Upload a stock CSV to run live inference</div>
          <div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#475569;'>
            Required columns: Date · Open · High · Low · Close · Volume
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 2: MODEL METRICS
# ─────────────────────────────────────────────────────────────────
with tab_metrics:

    metrics_df = None
    if metrics_csv is not None:
        try:
            metrics_df = pd.read_csv(metrics_csv)
        except Exception:
            pass

    if metrics_df is None:
        # Built-in metrics from the uploaded run
        metrics_df = pd.DataFrame([
            {"Fold": "1 ★", "MAPE_pct": 4.942, "R2": 0.9560, "DirAcc": 54.43},
            {"Fold": "2",   "MAPE_pct": 6.040, "R2": 0.8595, "DirAcc": 51.05},
            {"Fold": "3",   "MAPE_pct": 3.742, "R2": 0.8028, "DirAcc": 50.11},
            {"Fold": "4",   "MAPE_pct": 4.850, "R2": 0.9531, "DirAcc": 52.11},
            {"Fold": "TEST","MAPE_pct": 3.164, "R2": 0.8799, "DirAcc": 50.45},
        ])

    test_row   = metrics_df[metrics_df["Fold"].astype(str).str.upper() == "TEST"]
    fold_rows  = metrics_df[metrics_df["Fold"].astype(str).str.upper() != "TEST"]

    # Test set highlight cards
    if not test_row.empty:
        tr = test_row.iloc[0]
        st.markdown("<div class='section-title'>Hold-out Test Set Performance</div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        mape_val = float(tr["MAPE_pct"])
        r2_val   = float(tr["R2"])
        da_val   = float(tr["DirAcc"])

        mape_cls = "metric-good" if mape_val < 5 else ("metric-warn" if mape_val < 8 else "metric-bad")
        r2_cls   = "metric-good" if r2_val > 0.85 else ("metric-warn" if r2_val > 0.7 else "metric-bad")
        da_cls   = "metric-good" if da_val > 53 else ("metric-warn" if da_val > 50 else "metric-bad")

        with m1:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Test MAPE</div>
              <div class='metric-value {mape_cls}'>{mape_val:.3f}%</div>
              <div class='metric-sub'>Mean Absolute % Error</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Test R²</div>
              <div class='metric-value {r2_cls}'>{r2_val:.4f}</div>
              <div class='metric-sub'>Variance explained</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Directional Accuracy</div>
              <div class='metric-value {da_cls}'>{da_val:.2f}%</div>
              <div class='metric-sub'>vs 50% random baseline</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            edge = da_val - 50.0
            edge_cls = "metric-good" if edge > 3 else ("metric-warn" if edge > 0 else "metric-bad")
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>Edge over random</div>
              <div class='metric-value {edge_cls}'>+{edge:.2f}%</div>
              <div class='metric-sub'>Directional edge</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Walk-Forward Fold Comparison</div>", unsafe_allow_html=True)

    # Grouped bar chart
    folds_only = fold_rows.copy()
    fold_labels = folds_only["Fold"].astype(str).tolist()

    fig3 = make_subplots(rows=1, cols=3,
                         subplot_titles=["MAPE %", "R² Score", "Directional Accuracy %"])

    bar_colors = [ACCENT_BLUE if "★" not in str(f) else ACCENT_TEAL for f in fold_labels]

    fig3.add_trace(go.Bar(x=fold_labels, y=folds_only["MAPE_pct"],
                          marker_color=bar_colors, showlegend=False,
                          hovertemplate="%{x}: %{y:.3f}%<extra>MAPE</extra>"),
                   row=1, col=1)
    fig3.add_trace(go.Bar(x=fold_labels, y=folds_only["R2"],
                          marker_color=bar_colors, showlegend=False,
                          hovertemplate="%{x}: %{y:.4f}<extra>R²</extra>"),
                   row=1, col=2)
    fig3.add_trace(go.Bar(x=fold_labels, y=folds_only["DirAcc"],
                          marker_color=bar_colors, showlegend=False,
                          hovertemplate="%{x}: %{y:.2f}%<extra>DirAcc</extra>"),
                   row=1, col=3)

    # 50% line on dir acc
    fig3.add_hline(y=50, row=1, col=3, line_dash="dash",
                   line_color=ACCENT_AMBER, opacity=0.6, annotation_text="Random 50%",
                   annotation_font_color=ACCENT_AMBER)

    fig3.update_layout(**PLOT_LAYOUT, height=320,
                       showlegend=False,
                       paper_bgcolor="rgba(0,0,0,0)")
    fig3.update_annotations(font_size=11, font_color=TEXT_MUTED)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Raw table
    st.markdown("<div class='section-title'>Full Metrics Table</div>", unsafe_allow_html=True)
    display_df = metrics_df.copy()
    st.dataframe(
        display_df.style
            .format({"MAPE_pct": "{:.3f}%", "R2": "{:.4f}", "DirAcc": "{:.2f}%"})
            .set_properties(**{"font-family": "DM Mono, monospace", "font-size": "13px"})
            .applymap(lambda v: f"color:{ACCENT_GREEN}" if isinstance(v, float) and v > 0.85 else "",
                      subset=["R2"]),
        use_container_width=True,
        height=220,
    )


# ─────────────────────────────────────────────────────────────────
#  TAB 3: PRICE HISTORY
# ─────────────────────────────────────────────────────────────────
with tab_history:
    if df_raw is not None and not df_raw.empty:
        hist = df_raw.iloc[-hist_days:].copy()

        st.markdown("<div class='section-title'>OHLCV Candlestick Chart</div>", unsafe_allow_html=True)
        fig_c = go.Figure(go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"],
            low=hist["Low"],   close=hist["Close"],
            increasing_line_color=ACCENT_GREEN,
            decreasing_line_color=ACCENT_RED,
            name="OHLC",
        ))
        # 20-day MA
        ma20 = hist["Close"].rolling(20).mean()
        fig_c.add_trace(go.Scatter(x=hist.index, y=ma20, mode="lines",
                                   name="MA20", line=dict(color=ACCENT_AMBER, width=1.2, dash="dot")))
        fig_c.update_layout(**PLOT_LAYOUT, height=400,
                             xaxis_rangeslider_visible=False,
                             yaxis_title="Price (₹)")
        st.plotly_chart(fig_c, use_container_width=True, config={"displayModeBar": False})

        # Volume
        st.markdown("<div class='section-title'>Volume</div>", unsafe_allow_html=True)
        vol_colors = [ACCENT_GREEN if c >= o else ACCENT_RED
                      for c, o in zip(hist["Close"], hist["Open"])]
        fig_v = go.Figure(go.Bar(x=hist.index, y=hist["Volume"],
                                  marker_color=vol_colors, opacity=0.7, name="Volume"))
        fig_v.update_layout(**PLOT_LAYOUT, height=160, yaxis_title="Volume")
        st.plotly_chart(fig_v, use_container_width=True, config={"displayModeBar": False})

        # Returns distribution
        st.markdown("<div class='section-title'>Daily Return Distribution</div>", unsafe_allow_html=True)
        returns = hist["Close"].pct_change().dropna() * 100
        fig_r = go.Figure()
        fig_r.add_trace(go.Histogram(x=returns, nbinsx=60,
                                      marker_color=ACCENT_BLUE, opacity=0.8,
                                      name="Daily Return %"))
        fig_r.add_vline(x=float(returns.mean()), line_dash="dash",
                        line_color=ACCENT_TEAL, annotation_text=f"Mean {returns.mean():.3f}%",
                        annotation_font_color=ACCENT_TEAL)
        fig_r.update_layout(**PLOT_LAYOUT, height=250,
                             xaxis_title="Daily Return (%)", yaxis_title="Count")
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

    else:
        st.markdown("""
        <div style='background:#111827;border:1px solid #1E293B;border-radius:12px;
                    padding:3rem;text-align:center;'>
          <div style='font-size:2rem;margin-bottom:0.75rem'>📉</div>
          <div style='font-family:Syne,sans-serif;font-size:1rem;color:#94A3B8'>
            Upload a stock CSV in the sidebar to view price history
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 4: ARCHITECTURE
# ─────────────────────────────────────────────────────────────────
with tab_arch:
    if config:
        arch  = config.get("architecture", {})
        train = config.get("training", {})
        feat  = config.get("feature_cols", [])

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='section-title'>Model Architecture</div>", unsafe_allow_html=True)

            # Visual architecture diagram
            layers_info = [
                ("INPUT",    f"(60 timesteps × {config.get('n_features',13)} features)", ACCENT_BLUE),
                ("GRU 1",    f"{arch.get('gru1','?')} units · return_sequences=True", ACCENT_TEAL),
                ("LayerNorm","Stabilise activations across sequence", TEXT_MUTED),
                ("GRU 2",    f"{arch.get('gru2','?')} units · return_sequences=True", ACCENT_TEAL),
                ("LayerNorm","Stabilise activations across sequence", TEXT_MUTED),
                ("Attention",f"Bahdanau additive · {arch.get('attn_units','?')} units", ACCENT_AMBER),
                ("Dense 1",  f"{arch.get('dense1','?')} units · ReLU · Dropout {arch.get('dropout','?')}", ACCENT_BLUE),
                ("Dense 2",  f"{arch.get('dense2','?')} units · ReLU · Dropout {arch.get('dropout',0.15)*0.5:.3f}", ACCENT_BLUE),
                ("OUTPUT",   "1 unit · Log-return prediction", ACCENT_GREEN),
            ]

            for i, (name, desc, color) in enumerate(layers_info):
                arrow = "↓" if i < len(layers_info) - 1 else ""
                st.markdown(f"""
                <div style='display:flex;align-items:center;margin-bottom:2px;'>
                  <div style='background:{color}22;border:1px solid {color}55;border-radius:6px;
                              padding:0.4rem 0.9rem;min-width:120px;text-align:center;
                              font-family:DM Mono,monospace;font-size:0.75rem;color:{color};
                              font-weight:500;'>
                    {name}
                  </div>
                  <div style='margin-left:0.75rem;font-size:0.75rem;color:{TEXT_MUTED};
                              font-family:Inter,sans-serif;'>
                    {desc}
                  </div>
                </div>
                <div style='margin-left:55px;color:{TEXT_MUTED};font-size:0.9rem;line-height:0.8;margin-bottom:2px;'>
                  {arrow}
                </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-title'>Hyperparameters</div>", unsafe_allow_html=True)
            hp_items = [
                ("L2 Regularisation", f"{arch.get('l2', '?')}"),
                ("Recurrent Dropout", f"{arch.get('recurrent_dropout', '?')}"),
                ("Dropout",           f"{arch.get('dropout', '?')}"),
                ("Batch Size",        f"{train.get('batch_size', '?')}"),
                ("Max Epochs",        f"{train.get('max_epochs', '?')}"),
                ("Early Stop Pat.",   f"{train.get('early_stopping_patience', '?')}"),
                ("Init LR",           f"{train.get('init_lr', '?')}"),
                ("Peak LR",           f"{train.get('peak_lr', '?')}"),
                ("Warmup Epochs",     f"{train.get('warmup_epochs', '?')}"),
                ("Loss Function",     f"{train.get('loss', '?').title()}"),
                ("Kernel Init",       "glorot_uniform"),
                ("Sequence Length",   f"{config.get('sequence_len', '?')} days"),
            ]
            for k, v in hp_items:
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;align-items:center;
                            padding:0.45rem 1rem;border-bottom:1px solid {BORDER};
                            font-size:0.82rem;'>
                  <span style='color:{TEXT_MUTED};font-family:Inter,sans-serif;'>{k}</span>
                  <span style='font-family:DM Mono,monospace;color:{ACCENT_TEAL};'>{v}</span>
                </div>""", unsafe_allow_html=True)

        # Feature list
        st.markdown("<div class='section-title'>Input Features</div>", unsafe_allow_html=True)
        pills = "".join([
            f"<span class='arch-pill'>{f}</span>"
            for f in feat
        ])
        st.markdown(f"<div class='arch-grid'>{pills}</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-top:0.5rem;font-family:DM Mono,monospace;font-size:0.72rem;color:{TEXT_MUTED};'>
          Target: <span style='color:{ACCENT_AMBER}'>Log_Return</span>
          &nbsp;·&nbsp; Scaled with StandardScaler (fit on train only)
          &nbsp;·&nbsp; Features scaled with RobustScaler
        </div>""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown(f"""
    <div class='disclaimer'>
      ⚠️  <strong>Not financial advice.</strong> This dashboard is a research prototype.
      Predictions degrade rapidly beyond 5–7 days. Past model performance does not
      guarantee future results. Always consult a qualified financial advisor before
      making investment decisions.
    </div>""", unsafe_allow_html=True)
