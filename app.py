import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import requests
import os

# --- 1. CONFIG, SESSION STATE & CSS ---
st.set_page_config(page_title="AI Stock & SIP Terminal", page_icon="⚡", layout="wide")

if 'current_view' not in st.session_state:
    st.session_state.current_view = "Single Stock Analysis"
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = "RELIANCE"

if 'upward_picks' not in st.session_state:
    st.session_state.upward_picks = []
if 'downward_picks' not in st.session_state:
    st.session_state.downward_picks = []
if 'last_scanned_category' not in st.session_state:
    st.session_state.last_scanned_category = ""

if 'intraday_picks' not in st.session_state:
    st.session_state.intraday_picks = []

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = {}

def route_to_analysis(ticker):
    st.session_state.current_ticker = ticker
    st.session_state.current_view = "Single Stock Analysis"

def route_to_screener():
    st.session_state.current_view = "Market Screener"

st.markdown("""
<style>
    /* GLAMOROUS & NOISY BACKGROUND */
    .stApp { 
        background-color: #040506;
        background-image: 
            radial-gradient(at 15% 20%, rgba(16, 185, 129, 0.04) 0px, transparent 40%),
            radial-gradient(at 85% 15%, rgba(59, 130, 246, 0.05) 0px, transparent 40%),
            radial-gradient(at 50% 90%, rgba(239, 68, 68, 0.03) 0px, transparent 50%),
            url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.06'/%3E%3C/svg%3E");
        background-attachment: fixed;
    }
    .main { background: transparent; color: #E0E6ED; font-family: 'Inter', sans-serif; }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0b0d11; }
    ::-webkit-scrollbar-thumb { background: #2a2e39; border-radius: 4px; }
    
    div[data-testid="stMetric"] {
        background: rgba(11, 13, 17, 0.65);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 20px 25px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.8);
        border-color: rgba(255,255,255,0.15);
    }

    .signal-box {
        padding: 20px;
        border-radius: 12px;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .buy-signal { 
        background: rgba(16, 185, 129, 0.05); color: #10B981; 
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.05);
    }
    .sell-signal { 
        background: rgba(239, 68, 68, 0.05); color: #EF4444; 
        border: 1px solid rgba(239, 68, 68, 0.2);
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.05);
    }
    .neutral-signal { 
        background: rgba(245, 158, 11, 0.05); color: #F59E0B; 
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .reasoning-box {
        background-color: rgba(11, 13, 17, 0.7); 
        border-left: 4px solid #3B82F6; padding: 18px;
        border-radius: 0 8px 8px 0; margin-bottom: 15px; font-size: 0.95rem; color: #cbd5e1;
        backdrop-filter: blur(5px);
    }
    
    [data-testid="stSidebar"] { 
        background-color: rgba(11, 13, 17, 0.9); 
        border-right: 1px solid rgba(255,255,255,0.05); 
        backdrop-filter: blur(15px);
    }
    
    div.stButton > button {
        background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); color: #E0E6ED; 
        border-radius: 8px; padding: 6px 16px; transition: all 0.2s ease;
    }
    div.stButton > button:hover { 
        border-color: #10B981; color: #10B981; 
        box-shadow: 0 0 15px rgba(16,185,129,0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. ADVANCED DATA FETCHER & MATH ENGINE ---
@st.cache_data(ttl=300)
def get_stock_data(symbol, days_requested):
    clean_symbol = symbol.strip().upper()
    if "." not in clean_symbol: clean_symbol = f"{clean_symbol}.NS"
    total_days = max(days_requested * 2, 150)
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    df = yf.download(clean_symbol, period=f"{total_days}d", session=session, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df = df.xs(clean_symbol, level=1, axis=1)
    df = df.reset_index()
    if df['Date'].dt.tz is not None: df['Date'] = df['Date'].dt.tz_localize(None)
    
    df = df.dropna(subset=['Close'])
    if df.empty: return None
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# LIVE INTRADAY DATA FETCHER (5-Min Intervals)
@st.cache_data(ttl=60)
def get_intraday_data(symbol):
    clean_symbol = symbol.strip().upper()
    if "." not in clean_symbol: clean_symbol = f"{clean_symbol}.NS"
    
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    df = yf.download(clean_symbol, period="1d", interval="5m", session=session, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df = df.xs(clean_symbol, level=1, axis=1)
    df = df.reset_index()
    
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
    if df['Date'].dt.tz is not None: df['Date'] = df['Date'].dt.tz_localize(None)
    df = df.dropna(subset=['Close'])
    if df.empty or len(df) < 3: return None
    
    # Calculate VWAP
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    cum_vol = q.cumsum()
    df['VWAP'] = (p * q).cumsum() / cum_vol.replace(0, np.nan)
    df['VWAP'] = df['VWAP'].bfill().ffill()
    
    # Calculate short-term Intraday RSI (14 periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)
    
    return df

def calculate_advanced_indicators(df, forecast_win, analysis_win):
    df['SMA_Target'] = df['Close'].rolling(window=max(3, forecast_win)).mean()
    df['SMA_Base'] = df['Close'].rolling(window=max(5, analysis_win)).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean() 
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    df['R1'] = (2 * df['Pivot']) - df['Low'].shift(1)
    df['S1'] = (2 * df['Pivot']) - df['High'].shift(1)
    
    return df

def generate_monte_carlo_prediction(df, days_ahead, simulations=100):
    returns = df['Close'].pct_change().dropna()
    mu = returns.mean()
    vol = returns.std()
    last_price = float(df['Close'].iloc[-1])
    simulation_results = []
    
    for _ in range(simulations):
        price = last_price
        for _ in range(days_ahead):
            shock = np.random.normal(mu, vol)
            price = price * (1 + shock)
        simulation_results.append(price)
        
    predicted_price = np.median(simulation_results)
    pred_pct = ((predicted_price - last_price) / last_price) * 100
    upward_paths = [p for p in simulation_results if p > last_price]
    prob_up = (len(upward_paths) / simulations) * 100
    probable_up_pct = ((np.median(upward_paths) - last_price) / last_price) * 100 if upward_paths else 0.0
        
    return predicted_price, pred_pct, prob_up, probable_up_pct

def generate_algorithmic_reasoning(price, sma_target, sma_base, rsi, macd, macd_signal, bb_upper, bb_lower):
    reasons = []
    if price > sma_target and price > sma_base: reasons.append("✅ **Trend:** Price is dominating both short and macro moving averages. Buyers are in control.")
    elif price < sma_target and price < sma_base: reasons.append("❌ **Trend:** Price is trapped below key institutional baselines. Sellers are dominating.")
    else: reasons.append("⚠️ **Trend:** Mixed signals. Asset is consolidating between short and macro averages.")
        
    if macd > macd_signal: reasons.append("✅ **Momentum (MACD):** Upward momentum is accelerating (MACD crossover).")
    else: reasons.append("❌ **Momentum (MACD):** Downward momentum is active. Wait for momentum to flip before buying.")
        
    if rsi > 70: reasons.append("❌ **Risk (RSI):** Asset is mathematically overbought. High risk of immediate intraday pullback.")
    elif rsi < 30: reasons.append("✅ **Risk (RSI):** Asset is severely oversold. Potential bargain entry point.")
        
    if price >= bb_upper: reasons.append("⚠️ **Volatility:** Price is piercing the upper Bollinger Band. Reversal is highly probable today.")
    elif price <= bb_lower: reasons.append("⚠️ **Volatility:** Price hit the lower Bollinger Band. Expect an intraday bounce.")
        
    return reasons

def generate_t1_outlook(df):
    latest = df.iloc[-1]
    returns = df['Close'].pct_change().dropna()
    vol = returns.std()
    price = float(latest['Close'])
    
    bias = 0
    if price > float(latest['SMA_Target']): bias += 1
    else: bias -= 1
    if float(latest['MACD']) > float(latest['MACD_Signal']): bias += 1
    else: bias -= 1
    if float(latest['RSI']) > 70: bias -= 1
    elif float(latest['RSI']) < 30: bias += 1
    
    expected_move = vol * price
    
    if bias > 0:
        t1_target = price + expected_move
        direction = "🟢 BULLISH OPENING EXPECTED"
        reason = "Today's strong close and positive momentum structure suggest continued buying pressure tomorrow morning."
        color = "#10B981"
    elif bias < 0:
        t1_target = price - expected_move
        direction = "🔴 BEARISH OPENING EXPECTED"
        reason = "Weakness at today's close and negative momentum structure indicate a high probability of a lower open tomorrow."
        color = "#EF4444"
    else:
        t1_target = price
        direction = "🟡 NEUTRAL / CHOPPY OPEN"
        reason = "Conflicting intraday signals today. Expect a flat, unpredictable, or highly volatile open tomorrow."
        color = "#F59E0B"
        
    return t1_target, direction, reason, color

# GROQ AI ENGINE
def call_live_ai_model(prompt, ticker, price, rsi, macd_status, pivot, r1, s1, t1_dir, t1_target, probable_up_pct):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ **API Key Missing:** Please set your `GROQ_API_KEY` environment variable in your terminal to enable live AI responses."
    
    api_key = api_key.strip()
    
    system_context = (
        f"You are an expert AI stock trading assistant integrated into a professional terminal. "
        f"Current active stock: {ticker}. "
        f"Current Price: ₹{price:,.2f}, RSI: {rsi:.1f}, MACD Status: {macd_status}, "
        f"Daily Pivot: ₹{pivot:,.2f}, Resistance R1: ₹{r1:,.2f}, Support S1: ₹{s1:,.2f}, "
        f"T+1 Outlook: {t1_dir}, Probable Upward Gain: +{probable_up_pct:.2f}%. "
        f"Answer the user's question accurately. You can discuss this stock's data or answer general knowledge, coding, or life questions if they ask. Keep responses concise."
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_context},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as err:
        return f"⚠️ Error communicating with Groq AI model: HTTP {err.response.status_code} - {err.response.text}"
    except Exception as e:
        return f"⚠️ Error communicating with Groq AI model: {str(e)}"

# --- 3. TERMINAL UI SETUP ---
st.title("⚡ AI STOCK & SIP TERMINAL")
st.markdown("Professional Market Intelligence | Intraday Tactics, Screener & Investment Plans")
st.markdown("---")

st.sidebar.markdown("### 🧭 NAVIGATION")
view_options = ["Single Stock Analysis", "Live Intraday Tracker", "Market Screener", "Mutual Funds: SIP & Lump Sum (1-5 Mos)"]
current_index = view_options.index(st.session_state.current_view) if st.session_state.current_view in view_options else 0
selected_view = st.sidebar.radio("Go to:", view_options, index=current_index)
st.session_state.current_view = selected_view

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ TERMINAL CONTROL")
ticker_input = st.sidebar.text_input("Enter Ticker", st.session_state.current_ticker)
st.session_state.current_ticker = ticker_input.strip().upper()

time_period = st.sidebar.selectbox("Analysis Window (Macro Trend)", [30, 90, 180, 365], index=1)
prediction_days = st.sidebar.selectbox("Prediction Horizon (Micro Trend)", [7, 10, 15, 20, 30], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ ADVANCED OVERLAYS")
show_target_sma = st.sidebar.checkbox(f"Show {prediction_days}D Target Trend", value=True)
show_base_sma = st.sidebar.checkbox(f"Show {time_period}D Macro Trend", value=True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands (Volatility)", value=False)
show_macd = st.sidebar.checkbox("Show MACD Panel", value=True)


# ==========================================
# VIEW 1: SINGLE STOCK ANALYSIS
# ==========================================
if st.session_state.current_view == "Single Stock Analysis":
    st.button("⬅️ Back to Screener", on_click=route_to_screener)
    
    ticker = st.session_state.current_ticker
    if ticker:
        with st.spinner(f"Running Advanced Algorithmic Diagnostics for {ticker}..."):
            df_full = get_stock_data(ticker, days_requested=time_period)

        if df_full is not None and not df_full.empty:
            df_full = calculate_advanced_indicators(df_full, forecast_win=prediction_days, analysis_win=time_period)
            df_display = df_full.tail(time_period).copy()

            latest = df_display.iloc[-1]
            prev = df_display.iloc[-2]
            
            price = float(latest['Close'])
            prev_price = float(prev['Close'])
            change = price - prev_price
            pct_change = (change / prev_price) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("CURRENT PRICE (INR)", f"₹{price:,.2f}", f"₹{change:+.2f} ({pct_change:+.2f}%)")
            with col2: st.metric(f"MACRO TREND", f"₹{float(latest['SMA_Base']):,.2f}")
            with col3: st.metric("RSI MOMENTUM", f"{float(latest['RSI']):.1f}")
            with col4: st.metric("MACD SIGNAL", "BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH")

            st.markdown("---")
            st.subheader(f"📈 ADVANCED CHART & MACD OSCILLATOR")
            
            if show_macd:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
            else:
                fig = make_subplots(rows=1, cols=1)

            fig.add_trace(go.Candlestick(
                x=df_display['Date'], open=df_display['Open'], high=df_display['High'], 
                low=df_display['Low'], close=df_display['Close'], name="Price",
                increasing_line_color='#10B981', decreasing_line_color='#EF4444'
            ), row=1, col=1)
            
            if show_base_sma: fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['SMA_Base'], line=dict(color='#F59E0B', width=2), name="Macro Base"), row=1, col=1)
            if show_target_sma: fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['SMA_Target'], line=dict(color='#3B82F6', width=2, dash='dash'), name="Target Trend"), row=1, col=1)
            if show_bb:
                fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['BB_Upper'], line=dict(color='#94a3b8', width=1, dash='dot'), name="Upper Band"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['BB_Lower'], line=dict(color='#94a3b8', width=1, dash='dot'), name="Lower Band"), row=1, col=1)

            if show_macd:
                fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['MACD'], line=dict(color='#3B82F6', width=1.5), name="MACD"), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['MACD_Signal'], line=dict(color='#EF4444', width=1.5), name="Signal"), row=2, col=1)
                colors = ['#10B981' if val >= 0 else '#EF4444' for val in df_display['MACD_Hist']]
                fig.add_trace(go.Bar(x=df_display['Date'], y=df_display['MACD_Hist'], marker_color=colors, name="Histogram"), row=2, col=1)

            fig.update_layout(
                template="plotly_dark", 
                xaxis_rangeslider_visible=False, 
                height=850 if show_macd else 600,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)')
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', zerolinecolor='rgba(255,255,255,0.05)')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            
            col_diag, col_intra = st.columns([1.5, 1])
            
            with col_diag:
                st.subheader("🧠 ALGORITHMIC REASONING")
                
                is_short_bull = price > float(latest['SMA_Target'])
                is_long_bull = price > float(latest['SMA_Base'])
                rsi_val = float(latest['RSI'])
                
                if is_short_bull and is_long_bull and rsi_val < 70:
                    st.markdown('<div class="signal-box buy-signal"><b>🟢 EXECUTING BUY SIGNAL</b></div>', unsafe_allow_html=True)
                elif not is_short_bull and not is_long_bull:
                    st.markdown('<div class="signal-box sell-signal"><b>🔴 EXECUTING AVOID / SELL SIGNAL</b></div>', unsafe_allow_html=True)
                elif rsi_val > 70:
                    st.markdown('<div class="signal-box sell-signal"><b>🔴 REJECTED: OVERBOUGHT</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="signal-box neutral-signal"><b>🟡 HOLD: MIXED TRENDS</b></div>', unsafe_allow_html=True)

                st.markdown("**Why is the algorithm saying this?**")
                reasons = generate_algorithmic_reasoning(
                    price, float(latest['SMA_Target']), float(latest['SMA_Base']), 
                    rsi_val, float(latest['MACD']), float(latest['MACD_Signal']), 
                    float(latest['BB_Upper']), float(latest['BB_Lower'])
                )
                for reason in reasons: st.markdown(f'<div class="reasoning-box">{reason}</div>', unsafe_allow_html=True)

            with col_intra:
                st.subheader("🎯 INTRADAY TACTICS (TODAY)")
                pivot = float(latest['Pivot'])
                r1 = float(latest['R1'])
                s1 = float(latest['S1'])
                
                st.markdown(f"""
<div style="background-color: rgba(11,13,17,0.7); backdrop-filter: blur(5px); padding: 25px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05);">
<h4 style="color: #E0E6ED; margin-top:0; font-size: 1.1rem;">Capital Protection Plan</h4>
<p style="color: #94a3b8; font-size: 0.9rem;">Use these automated support/resistance floor levels to prevent losses within the trading day.</p>
<div style="margin-top: 20px;">
<span style="color: #10B981; font-weight: 700;">↑ Breakout Target (R1):</span><br>
<span style="font-size: 1.4rem; font-weight: 800; color: #E0E6ED;">₹{r1:,.2f}</span>
</div>
<div style="margin-top: 20px;">
<span style="color: #3B82F6; font-weight: 700;">⟷ Daily Pivot (Eq):</span><br>
<span style="font-size: 1.4rem; font-weight: 800; color: #E0E6ED;">₹{pivot:,.2f}</span>
</div>
<div style="margin-top: 20px;">
<span style="color: #EF4444; font-weight: 700;">↓ Stop-Loss (S1):</span><br>
<span style="font-size: 1.4rem; font-weight: 800; color: #E0E6ED;">₹{s1:,.2f}</span>
</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            
            st.subheader(f"🔮 {prediction_days}-DAY MONTE CARLO FORECAST")
            predicted_price, pred_pct, prob_up, probable_up_pct = generate_monte_carlo_prediction(df_full, days_ahead=prediction_days, simulations=100)
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1: st.metric(f"MEDIAN TARGET (T+{prediction_days})", f"₹{predicted_price:,.2f}", f"{pred_pct:+.2f}%")
            with col_p2: st.metric("UPWARD PROBABILITY", f"{prob_up:.1f}%")
            with col_p3: st.metric("PROBABLE UPWARD GAIN", f"+{probable_up_pct:.2f}%")
            
            st.markdown('<div style="color: #94a3b8; font-size: 0.85rem; padding-top:10px; margin-bottom: 20px;"><i>Model isolates positive simulation outcomes to calculate the median probable upward gain percentage.</i></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("⏱️ TOMORROW'S OUTLOOK (T+1)")
            t1_target, t1_direction, t1_reason, t1_color = generate_t1_outlook(df_full)
            
            col_t1_metrics, col_t1_context = st.columns([1, 2])
            with col_t1_metrics:
                st.markdown(f"""
<div style="background-color: rgba(11,13,17,0.7); backdrop-filter: blur(5px); padding: 25px; border-radius: 16px; border: 1px solid {t1_color}40; box-shadow: 0 5px 20px {t1_color}10;">
<div style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 8px;">T+1 Expected Open</div>
<div style="color: #E0E6ED; font-size: 2rem; font-weight: 800; margin-bottom: 12px;">₹{t1_target:,.2f}</div>
<div style="color: {t1_color}; font-weight: 800; font-size: 1.15rem; letter-spacing: 0.5px;">{t1_direction}</div>
</div>
""", unsafe_allow_html=True)
                
            with col_t1_context:
                st.markdown(f"""
<div style="padding-top: 15px; padding-left: 10px;">
<h4 style="color: #E0E6ED; margin-top:0; font-size: 1.2rem;">Based on Today's Market Conditions</h4>
<p style="color: #cbd5e1; font-size: 1.1rem; line-height: 1.7;">
{t1_reason}
</p>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")

            st.subheader("🛡️ POSITION ADVISORY: HOLD, SELL, OR WAIT?")
            
            if is_short_bull and is_long_bull and rsi_val < 70 and float(latest['MACD']) > float(latest['MACD_Signal']):
                holding_advice = "🟢 **RECOMMENDATION: HOLD YOUR POSITION / RIDE THE TREND**"
                advice_color = "#10B981"
                advice_bg = "rgba(16, 185, 129, 0.08)"
                detailed_action = "Your shares are backed by solid upward momentum and institutional trendlines. **Do not sell yet.** Keep holding to maximize gains, but trail your stop-loss upward to protect your profits."
            elif rsi_val > 70 or price < float(latest['S1']):
                holding_advice = "🔴 **RECOMMENDATION: SQUARE OFF / SELL IMMEDIATELY**"
                advice_color = "#EF4444"
                advice_bg = "rgba(239, 68, 68, 0.08)"
                detailed_action = "The asset is either heavily overbought or has breached key emergency support levels. **Lock in your profits or cut losses now** to prevent capital erosion from a sudden correction."
            else:
                holding_advice = "🟡 **RECOMMENDATION: WAIT & WATCH FOR THE NEXT FEW DAYS**"
                advice_color = "#F59E0B"
                advice_bg = "rgba(245, 158, 11, 0.08)"
                detailed_action = "The stock is currently consolidating or displaying mixed signals between short and long trends. **Do not panic sell or buy more.** Observe price action over the next few sessions until a clear breakout occurs."

            st.markdown(f"""
<div style="background-color: {advice_bg}; border: 1px solid {advice_color}50; padding: 25px; border-radius: 16px; backdrop-filter: blur(5px);">
    <div style="color: {advice_color}; font-size: 1.25rem; font-weight: 800; margin-bottom: 12px;">{holding_advice}</div>
    <p style="color: #E0E6ED; font-size: 1.1rem; line-height: 1.6; margin: 0;">
        {detailed_action}
    </p>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")

            st.subheader(f"💬 AI ASSISTANT CHAT: ASK ANYTHING ABOUT {ticker} OR GENERAL TOPICS")
            st.write("Powered by Groq AI. Ask about this stock, market trends, coding, or anything else!")

            if ticker not in st.session_state.chat_messages:
                st.session_state.chat_messages[ticker] = [
                    {"role": "assistant", "content": f"Hello! I am your live AI analyst. I have loaded the data for **{ticker}**. Feel free to ask me anything!"}
                ]

            for message in st.session_state.chat_messages[ticker]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_query = st.chat_input(f"Ask anything...")
            if user_query:
                st.session_state.chat_messages[ticker].append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                macd_status = "BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "BEARISH"
                
                with st.spinner("AI is thinking..."):
                    ai_reply = call_live_ai_model(
                        user_query, ticker, price, float(latest['RSI']), macd_status, 
                        float(latest['Pivot']), float(latest['R1']), float(latest['S1']), t1_direction, t1_target, probable_up_pct
                    )

                st.session_state.chat_messages[ticker].append({"role": "assistant", "content": ai_reply})
                with st.chat_message("assistant"):
                    st.markdown(ai_reply)

        else:
            st.error("Could not fetch valid pricing data. The stock may be unlisted, or there is missing market data for this ticker.")

# ==========================================
# VIEW 2: LIVE INTRADAY TRACKER & SCANNER
# ==========================================
elif st.session_state.current_view == "Live Intraday Tracker":
    st.subheader(f"📡 Live Intraday Tracker: {st.session_state.current_ticker}")
    st.write("Real-time 5-minute interval tracking with dynamic VWAP and momentum scoring.")
    
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()
        
    ticker = st.session_state.current_ticker
    if ticker:
        with st.spinner(f"Fetching live 5-minute market data for {ticker}..."):
            df_intra = get_intraday_data(ticker)
            
        if df_intra is not None and not df_intra.empty:
            latest_i = df_intra.iloc[-1]
            start_price = float(df_intra.iloc[0]['Open'])
            cur_price = float(latest_i['Close'])
            vwap_val = float(latest_i['VWAP'])
            rsi_val = float(latest_i['RSI'])
            
            day_high = float(df_intra['High'].max())
            day_low = float(df_intra['Low'].min())
            net_change = cur_price - start_price
            net_pct = (net_change / start_price) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("LIVE PRICE", f"₹{cur_price:,.2f}", f"{net_pct:+.2f}%")
            with c2: st.metric("VWAP (Anchor)", f"₹{vwap_val:,.2f}")
            with c3: st.metric("DAY HIGH", f"₹{day_high:,.2f}")
            with c4: st.metric("DAY LOW", f"₹{day_low:,.2f}")
            
            st.markdown("---")
            
            # Live Signal Logic
            if cur_price > vwap_val and rsi_val < 70:
                signal = "🟢 INTRADAY BUY / GO LONG"
                box_class = "buy-signal"
                msg = f"Price is **above VWAP** and momentum is solid (RSI: {rsi_val:.1f}). Buyers are in control today."
            elif cur_price < vwap_val and rsi_val > 30:
                signal = "🔴 INTRADAY SELL / GO SHORT"
                box_class = "sell-signal"
                msg = f"Price is **below VWAP** and momentum is weak (RSI: {rsi_val:.1f}). Sellers are dominating."
            else:
                signal = "🟡 NEUTRAL / HOLD"
                box_class = "neutral-signal"
                msg = f"Mixed signals. Price is hovering near VWAP or RSI is stretched ({rsi_val:.1f}). Wait for a clear breakout."
                
            st.markdown(f"""
            <div class="signal-box {box_class}" style="background-color: rgba(11,13,17,0.8); backdrop-filter: blur(5px);">
                <div style="font-size: 1.3rem; font-weight: 800; margin-bottom: 8px;">{signal}</div>
                <div style="color: #cbd5e1; font-size: 1.05rem;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            
            fig_i = go.Figure()
            fig_i.add_trace(go.Candlestick(
                x=df_intra['Date'], open=df_intra['Open'], high=df_intra['High'], 
                low=df_intra['Low'], close=df_intra['Close'], name="Price",
                increasing_line_color='#10B981', decreasing_line_color='#EF4444'
            ))
            fig_i.add_trace(go.Scatter(x=df_intra['Date'], y=df_intra['VWAP'], line=dict(color='#F59E0B', width=2), name="VWAP"))
            
            fig_i.update_layout(
                template="plotly_dark", xaxis_rangeslider_visible=False, height=500,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_i, use_container_width=True)
        else:
            st.warning("Live intraday data is currently unavailable for this ticker. The market may be closed or the ticker does not support 5-minute intervals.")
            
    # ==========================================
    # --- MULTI-STOCK INTRADAY BREAKOUT SCANNER ---
    # ==========================================
    st.markdown("---")
    st.subheader("⚡ Live Intraday Buy Recommendations (Multi-Price Tiers)")
    st.write("Scan over 60 liquid stocks across NSE spanning all budget segments (Under ₹100 up to ₹3,000+) to surface top intraday breakouts.")

    # Price Limit Filter Selector
    price_filter = st.selectbox(
        "Filter by Max Stock Price (INR):",
        ["All Prices", "Under ₹100", "Under ₹200", "Under ₹500", "Under ₹1,000", "Under ₹2,000", "Under ₹3,000"],
        index=0
    )

    # 60+ High-Volume Multi-Tier Watchlist across NSE
    MULTI_TIER_WATCHLIST = [
        # Under 100
        "SUZLON", "IDEA", "YESBANK", "RPOWER", "GTLINFRA", "NHPC", "IOB", "UCOBANK", "CENTRALBK",
        # Under 200
        "IDFCFIRSTB", "BANKINDIA", "UNIONBANK", "SAIL", "NATIONALUM", "TATASTEEL", "BHEL", "NBCC", "ZOMATO",
        # Under 500
        "IRFC", "RVNL", "PFC", "RECLTD", "IOC", "BPCL", "GAIL", "CANBK", "PNB", "ONGC", "EXIDEIND", "TATAPOWER", "MOTHERSON",
        # Under 1000
        "SBIN", "TATAMOTORS", "ITC", "WIPRO", "HINDALCO", "COALINDIA", "BEL", "HAL", "JSWSTEEL", "DLF", "CUMMINSIND", "DABUR",
        # Under 2000
        "HDFCBANK", "ICICIBANK", "AXISBANK", "INFY", "BHARTIARTL", "KOTAKBANK", "M&M", "SUNPHARMA", "HCLTECH", "TECHM", "CIPLA",
        # Under 3000 & Large Caps
        "RELIANCE", "TCS", "LT", "BAJFINANCE", "ASIANPAINT", "TITAN", "MARUTI", "ULTRACEMCO"
    ]

    if st.button("🚀 Scan 60+ Stocks for Intraday Buy Setups"):
        scanned_results = []
        prog = st.progress(0)
        
        for i, sym in enumerate(MULTI_TIER_WATCHLIST):
            prog.progress((i + 1) / len(MULTI_TIER_WATCHLIST))
            df_s = get_intraday_data(sym)
            
            if df_s is not None and not df_s.empty:
                last_row = df_s.iloc[-1]
                o_price = float(df_s.iloc[0]['Open'])
                c_price = float(last_row['Close'])
                vwap_p = float(last_row['VWAP'])
                rsi_p = float(last_row['RSI'])
                
                day_gain = ((c_price - o_price) / o_price) * 100
                vwap_diff = ((c_price - vwap_p) / vwap_p) * 100
                
                # Balanced Intraday Filter: Above VWAP and Positive Momentum
                if c_price > vwap_p and rsi_p >= 48 and day_gain > -0.5:
                    scanned_results.append({
                        "Symbol": sym,
                        "Price": c_price,
                        "VWAP": vwap_p,
                        "RSI": rsi_p,
                        "Gain": day_gain,
                        "VWAP_Margin": vwap_diff
                    })
                    
        prog.empty()
        st.session_state.intraday_picks = sorted(scanned_results, key=lambda x: x['Gain'], reverse=True)

    # Filter & Display Logic
    if st.session_state.intraday_picks:
        picks = st.session_state.intraday_picks
        
        # Apply Price Bracket Filters
        if price_filter == "Under ₹100":
            filtered_picks = [p for p in picks if p['Price'] <= 100]
        elif price_filter == "Under ₹200":
            filtered_picks = [p for p in picks if p['Price'] <= 200]
        elif price_filter == "Under ₹500":
            filtered_picks = [p for p in picks if p['Price'] <= 500]
        elif price_filter == "Under ₹1,000":
            filtered_picks = [p for p in picks if p['Price'] <= 1000]
        elif price_filter == "Under ₹2,000":
            filtered_picks = [p for p in picks if p['Price'] <= 2000]
        elif price_filter == "Under ₹3,000":
            filtered_picks = [p for p in picks if p['Price'] <= 3000]
        else:
            filtered_picks = picks

        st.markdown(f"### 🟢 Found **{len(filtered_picks)}** Intraday Buy Recommendations ({price_filter})")
        
        if filtered_picks:
            # Render Cards in 3-Column Grid
            cols = st.columns(3)
            for idx, item in enumerate(filtered_picks):
                col = cols[idx % 3]
                with col:
                    st.markdown(f"""
                    <div style="background-color: rgba(11,13,17,0.75); backdrop-filter: blur(8px); padding: 18px; border-radius: 14px; border: 1px solid rgba(16, 185, 129, 0.25); margin-bottom: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 1.15rem; font-weight: 800; color: #E0E6ED;">{item['Symbol']}</span>
                            <span style="background: rgba(16, 185, 129, 0.15); color: #10B981; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.85rem;">+{item['Gain']:.2f}%</span>
                        </div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #E0E6ED; margin-bottom: 6px;">₹{item['Price']:,.2f}</div>
                        <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">
                            VWAP: <b style="color: #F59E0B;">₹{item['VWAP']:,.2f}</b> (+{item['VWAP_Margin']:.2f}% above)<br>
                            RSI Momentum: <b style="color: #3B82F6;">{item['RSI']:.1f}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.button(f"Analyze {item['Symbol']}", key=f"scan_btn_{item['Symbol']}", on_click=route_to_analysis, args=(item["Symbol"],))
        else:
            st.info(f"No active intraday breakout setups found strictly under the '{price_filter}' limit. Try selecting 'All Prices'.")
    else:
        st.info("Click 'Scan 60+ Stocks for Intraday Buy Setups' above to generate top real-time intraday recommendations.")


# ==========================================
# VIEW 3: AI MARKET SCREENER
# ==========================================
elif st.session_state.current_view == "Market Screener":
    st.subheader("🔥 Advanced Market Screener")
    st.write("Scan entire asset categories to find mathematically optimal setups.")
    
    screener_category = st.selectbox(
        "Select Asset Category", 
        ["Nifty 100 (Top Indian Stocks)", "Indian ETFs (Baskets & Proxies)", "High-Growth Midcaps"],
        key="screener_category_selectbox"
    )
    
    NIFTY_100 = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS", "ITC", "LT", "AXISBANK", "BAJFINANCE", "BHARTIARTL", "KOTAKBANK", "HINDUNILVR", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "HCLTECH", "M&M", "POWERGRID", "NTPC", "TATASTEEL", "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BAJAJ-AUTO", "BAJAJFINSV", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "JSWSTEEL", "LTIM", "NESTLEIND", "ONGC", "SBILIFE", "TATACOMM", "TATACONSUM", "TECHM", "UPL", "AMBUJACEM", "BANKBARODA", "BEL", "BOSCHLTD", "CANBK", "CHOLAFIN", "COLPAL", "DLF", "GAIL", "GODREJCP", "HAL", "HAVELLS", "ICICIGI", "ICICIPRULI", "IDFCFIRSTB", "INDIGO", "IOC", "IRCTC", "JINDALSTEL", "MARICO", "MUTHOOTFIN", "NAUKRI", "PIDILITIND", "PIIND", "PNB", "RECLTD", "SHREECEM", "SIEMENS", "SRF", "TVSMOTOR", "UNITEDSPR", "VEDL", "ZOMATO"]
    INDIAN_ETFS = ["NIFTYBEES", "BANKBEES", "GOLDBEES", "LIQUIDBEES", "ITBEES", "PHARMABEES", "JUNIORBEES", "MID150BEES", "MON100", "CPSEETF", "HDFCNIFTY", "SBIETFIT"]
    MIDCAP_GROWTH = ["ABBOTINDIA", "AUBANK", "BANDHANBNK", "CUMMINSIND", "FEDERALBNK", "IDBI", "LUPIN", "MRF", "OFSS", "PAGEIND", "PERSISTENT", "POLYCAB", "TRENT", "VOLTAS", "ZEEL"]
    
    if screener_category == "Nifty 100 (Top Indian Stocks)": scan_list = NIFTY_100
    elif screener_category == "Indian ETFs (Baskets & Proxies)": scan_list = INDIAN_ETFS
    else: scan_list = MIDCAP_GROWTH
    
    if st.button(f"🚀 Run Scan on {screener_category}"):
        upward_picks = []
        downward_picks = []
        progress_bar = st.progress(0)
        
        for i, sym in enumerate(scan_list):
            progress_bar.progress((i + 1) / len(scan_list))
            df_scan = get_stock_data(sym, days_requested=10)
            if df_scan is not None and not df_scan.empty:
                df_scan = calculate_advanced_indicators(df_scan, forecast_win=prediction_days, analysis_win=time_period)
                last_row = df_scan.iloc[-1]
                
                price = float(last_row['Close'])
                sma_base = float(last_row['SMA_Base'])
                sma_target = float(last_row['SMA_Target'])
                rsi = float(last_row['RSI'])
                
                if price > sma_base and price > sma_target and 35 < rsi < 65:
                    _, _, _, probable_up_pct = generate_monte_carlo_prediction(df_scan, days_ahead=prediction_days, simulations=30)
                    upward_picks.append({"Symbol": sym, "Price": price, "RSI": rsi, "Gain": probable_up_pct})
                elif price < sma_base or price < sma_target or rsi > 70:
                    downward_picks.append({"Symbol": sym, "Price": price, "RSI": rsi})
                    
        progress_bar.empty()
        
        st.session_state.upward_picks = sorted(upward_picks, key=lambda x: x['Gain'], reverse=True)
        st.session_state.downward_picks = downward_picks
        st.session_state.last_scanned_category = screener_category

    if st.session_state.upward_picks or st.session_state.downward_picks:
        st.markdown(f"### Results for: {st.session_state.last_scanned_category}")
        col_up, col_down = st.columns(2)
        
        with col_up:
            st.markdown(f"### 🟢 Upward Trend ({len(st.session_state.upward_picks)} Found)")
            if st.session_state.upward_picks:
                for pick in st.session_state.upward_picks:
                    with st.container():
                        st.markdown(f'''
<div class="signal-box buy-signal" style="background: rgba(11,13,17,0.7); backdrop-filter: blur(5px);">
    <b>{pick["Symbol"]}</b> — ₹{pick["Price"]:,.2f}<br>
    <small>RSI: {pick["RSI"]:.1f} | <span style="color: #10B981; font-weight: bold;">Est. Return if Bought Now: +{pick["Gain"]:.2f}%</span></small>
</div>
''', unsafe_allow_html=True)
                        st.button(f"Analyze {pick['Symbol']}", key=f"buy_{pick['Symbol']}", on_click=route_to_analysis, args=(pick["Symbol"],))
            else:
                st.info("No strong upward setups found in last scan.")
                
        with col_down:
            st.markdown(f"### 🔴 Downward Trend ({len(st.session_state.downward_picks)} Found)")
            if st.session_state.downward_picks:
                for pick in st.session_state.downward_picks:
                    with st.container():
                        st.markdown(f'''
<div class="signal-box sell-signal" style="background: rgba(11,13,17,0.7); backdrop-filter: blur(5px);">
    <b>{pick["Symbol"]}</b> — ₹{pick["Price"]:,.2f}<br>
    <small>RSI: {pick["RSI"]:.1f}</small>
</div>
''', unsafe_allow_html=True)
                        st.button(f"Analyze {pick['Symbol']}", key=f"sell_{pick['Symbol']}", on_click=route_to_analysis, args=(pick["Symbol"],))
            else:
                st.info("No downward threats detected in last scan.")
    else:
        st.info("Select a category above and click 'Run Scan' to generate recommendations.")


# ==========================================
# VIEW 4: MUTUAL FUNDS (SIP & LUMP SUM)
# ==========================================
elif st.session_state.current_view == "Mutual Funds: SIP & Lump Sum (1-5 Mos)":
    st.subheader("💰 Mutual Funds: Best SIP & Lump Sum Investment Plans")
    st.write("Explore top-performing mutual fund schemes. Choose your investment mode (Monthly SIP vs. One-Time Lump Sum) and select your contribution amount in exact multiples of ₹500 (from ₹500 to ₹5,000) to view detailed 1 to 5 month projected portfolio growth.")
    
    inv_mode = st.radio("Select Investment Mode:", ["Monthly SIP (Recurring)", "One-Time Lump Sum"], horizontal=True)
    
    amounts = [500 * i for i in range(1, 11)]
    selected_amount = st.selectbox(f"Select {inv_mode} Investment Amount (INR):", amounts, index=1)
    
    top_funds = [
        {"name": "ICICI Prudential Bluechip Fund", "category": "Large Cap Equity", "risk": "Moderate-High", "monthly_rate": 0.012},
        {"name": "Nippon India Large Cap Fund", "category": "Large Cap Equity", "risk": "Moderate-High", "monthly_rate": 0.013},
        {"name": "HDFC Flexi Cap Fund", "category": "Flexi Cap Equity", "risk": "High", "monthly_rate": 0.015},
        {"name": "Quant Flexi Cap Fund", "category": "Flexi Cap Equity", "risk": "Very High", "monthly_rate": 0.017},
        {"name": "Edelweiss Flexi Cap Fund", "category": "Flexi Cap Equity", "risk": "High", "monthly_rate": 0.014},
        {"name": "Motilal Oswal Midcap Fund", "category": "Mid Cap Equity", "risk": "Very High", "monthly_rate": 0.022},
        {"name": "Nippon India Growth Mid Cap", "category": "Mid Cap Equity", "risk": "Very High", "monthly_rate": 0.020},
        {"name": "HSBC Midcap Fund", "category": "Mid Cap Equity", "risk": "Very High", "monthly_rate": 0.021},
        {"name": "Quant Small Cap Fund", "category": "Small Cap Equity", "risk": "Very High", "monthly_rate": 0.024},
        {"name": "Nippon India Small Cap Fund", "category": "Small Cap Equity", "risk": "Very High", "monthly_rate": 0.022},
        {"name": "Bandhan Small Cap Fund", "category": "Small Cap Equity", "risk": "Very High", "monthly_rate": 0.023},
        {"name": "ICICI Prudential Equity & Debt", "category": "Aggressive Hybrid", "risk": "Moderate", "monthly_rate": 0.012},
        {"name": "HDFC Balanced Advantage Fund", "category": "Balanced Advantage", "risk": "Moderate", "monthly_rate": 0.011},
        {"name": "ICICI Prudential Nifty Next 50", "category": "Index Fund", "risk": "High", "monthly_rate": 0.016},
        {"name": "SBI PSU Direct Plan Growth", "category": "Sectoral / Thematic", "risk": "Very High", "monthly_rate": 0.025}
    ]
    
    st.markdown(f"### 📊 Projections for {inv_mode}: **₹{selected_amount:,}**")
    st.markdown("---")
    
    for fund in top_funds:
        rate = fund["monthly_rate"]
        vals = []
        
        if "Monthly SIP" in inv_mode:
            current_val = 0
            for m in range(1, 6):
                current_val = (current_val + selected_amount) * (1 + rate)
                vals.append(current_val)
            total_invested = selected_amount * 5
        else:
            current_val = float(selected_amount)
            for m in range(1, 6):
                current_val = current_val * (1 + rate)
                vals.append(current_val)
            total_invested = selected_amount
            
        m1, m2, m3, m4, m5 = vals
        total_gain = m5 - total_invested
        gain_pct = (total_gain / total_invested) * 100
        
        card_html = f"""<div style="background-color: rgba(11,13,17,0.75); backdrop-filter: blur(8px); padding: 22px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
<h3 style="color: #E0E6ED; margin: 0; font-size: 1.2rem;">🌟 {fund['name']}</h3>
<span style="background: rgba(59, 130, 246, 0.1); color: #3B82F6; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: bold;">{fund['category']}</span>
</div>
<div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 15px;">
Risk Level: <b style="color: #F59E0B;">{fund['risk']}</b> | Mode: <b>{inv_mode}</b> (₹{selected_amount:,})
</div>
<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 10px; text-align: center;">
<div>
<div style="color: #94a3b8; font-size: 0.75rem;">1 Month</div>
<div style="color: #10B981; font-weight: bold; font-size: 0.95rem;">₹{m1:,.0f}</div>
</div>
<div>
<div style="color: #94a3b8; font-size: 0.75rem;">2 Months</div>
<div style="color: #10B981; font-weight: bold; font-size: 0.95rem;">₹{m2:,.0f}</div>
</div>
<div>
<div style="color: #94a3b8; font-size: 0.75rem;">3 Months</div>
<div style="color: #10B981; font-weight: bold; font-size: 0.95rem;">₹{m3:,.0f}</div>
</div>
<div>
<div style="color: #94a3b8; font-size: 0.75rem;">4 Months</div>
<div style="color: #10B981; font-weight: bold; font-size: 0.95rem;">₹{m4:,.0f}</div>
</div>
<div>
<div style="color: #94a3b8; font-size: 0.75rem;">5 Months</div>
<div style="color: #3B82F6; font-weight: bold; font-size: 1.05rem;">₹{m5:,.0f}</div>
</div>
</div>
<div style="margin-top: 12px; font-size: 0.85rem; color: #cbd5e1; display: flex; justify-content: space-between;">
<span>Total Capital Committed: <b>₹{total_invested:,}</b></span>
<span style="color: #10B981; font-weight: bold;">Projected 5-Month Return: +{gain_pct:.2f}% (₹{total_gain:,.0f})</span>
</div>
</div>"""
        st.markdown(card_html, unsafe_allow_html=True)
