"""
Crypto Dashboard — Streamlit MVP
Single-file Streamlit app (app.py) to monitor BTC, ETH, XRP and TRX
- Fetches OHLCV from Binance (via ccxt)
- Optional: fetch funding rates from Binance Futures REST
- Computes technical indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, OBV, VWAP)
- Produces rule-based buy/sell/hold recommendations and daily highlights
- Stores daily summaries in a local SQLite DB

Requirements (install into a virtualenv):
pip install streamlit ccxt pandas numpy requests ta plotly streamlit-autorefresh

How to run locally:
1. Save this file as app.py
2. streamlit run app.py

Notes for extension:
- Add API keys as environment variables for Etherscan/Deribit/TronGrid if you want on-chain or derivatives features.
- For continuous production updates, run this behind a small backend (FastAPI) that writes to InfluxDB/TimescaleDB and serve the frontend from Streamlit/React.

"""

import os
import time
import sqlite3
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import requests
import ccxt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import plotly.graph_objects as go

# Optional autorefresh component (install streamlit-autorefresh)
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORELOAD_AVAILABLE = True
except Exception:
    AUTORELOAD_AVAILABLE = False

# ----------------- Config -----------------
SYMBOLS = {
    'Bitcoin': 'BTC/USDT',
    'Ethereum': 'ETH/USDT',
    'XRP': 'XRP/USDT',
    'Tron': 'TRX/USDT'
}

DB_FILE = 'crypto_dashboard.db'
BINANCE_FUTURES_FUNDING_ENDPOINT = 'https://fapi.binance.com/fapi/v1/fundingRate'

# ----------------- Utils ------------------

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT,
            ts_utc TEXT,
            price REAL,
            signal TEXT,
            change_24h REAL,
            volume REAL
        )
    ''')
    conn.commit()
    conn.close()


@st.cache_data(ttl=30)
def fetch_ohlcv_binance(symbol: str, timeframe: str = '1m', limit: int = 500):
    """Fetch OHLCV from Binance via ccxt (spot). Cached for 30s to reduce rate usage."""
    exchange = ccxt.binance({'enableRateLimit': True})
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        st.error(f"Error fetching OHLCV: {e}")
        return None
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def fetch_latest_funding_rate_binance(symbol: str):
    """Get the most recent funding rate for a perpetual on Binance Futures (symbol like BTCUSDT). Returns float or None."""
    try:
        params = {'symbol': symbol.replace('/', ''), 'limit': 1}
        r = requests.get(BINANCE_FUTURES_FUNDING_ENDPOINT, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            return float(data[-1]['fundingRate'])
    except Exception as e:
        # silently ignore; just return None
        return None
    return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to df (returns a new df)."""
    out = df.copy()
    close = out['close']
    high = out['high']
    low = out['low']
    vol = out['volume']

    out['sma20'] = SMAIndicator(close, window=20, fillna=True).sma_indicator()
    out['sma50'] = SMAIndicator(close, window=50, fillna=True).sma_indicator()
    out['sma200'] = SMAIndicator(close, window=200, fillna=True).sma_indicator()

    out['ema9'] = EMAIndicator(close, window=9, fillna=True).ema_indicator()
    out['ema21'] = EMAIndicator(close, window=21, fillna=True).ema_indicator()
    out['ema50'] = EMAIndicator(close, window=50, fillna=True).ema_indicator()

    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out['macd'] = macd.macd()
    out['macd_signal'] = macd.macd_signal()
    out['macd_hist'] = macd.macd_diff()

    out['rsi14'] = RSIIndicator(close, window=14, fillna=True).rsi()

    bb = BollingerBands(close, window=20, window_dev=2, fillna=True)
    out['bb_h'] = bb.bollinger_hband()
    out['bb_l'] = bb.bollinger_lband()

    out['atr14'] = AverageTrueRange(high=high, low=low, close=close, window=14, fillna=True).average_true_range()

    obv = OnBalanceVolumeIndicator(close=close, volume=vol, fillna=True)
    out['obv'] = obv.on_balance_volume()

    # simple vwap over the whole df
    tp = (out['high'] + out['low'] + out['close']) / 3
    out['vwap'] = (tp * out['volume']).cumsum() / out['volume'].cumsum()

    return out


def generate_signal(latest: pd.Series) -> str:
    """Simple rule-based composite signal: BUY / SELL / HOLD / CAUTION (explanatory text)."""
    close = latest['close']
    ema21 = latest['ema21']
    ema50 = latest['ema50']
    macd_hist = latest['macd_hist']
    rsi = latest['rsi14']
    vol = latest['volume']
    sma50 = latest['sma50']

    bull = (close > ema21) and (ema21 > ema50) and (macd_hist > 0) and (rsi < 75) and (vol > 0)
    bear = (close < ema21) and (ema21 < ema50) and (macd_hist < 0) and (rsi > 25)

    if bull and rsi < 60:
        return 'BUY — tendencia alcista con momentum. Confirmar con volumen y gestión de riesgo.'
    if bull and rsi >= 60:
        return 'CAUTION — sobrecompra cercana (RSI alto). Esperar retroceso o reducir tamaño.'
    if bear:
        return 'SELL — tendencia bajista. Evitar entradas largas o preparar stop.'
    return 'HOLD — sin señal clara. Esperar confirmación (price/volume).' 


def daily_highlights(df: pd.DataFrame) -> dict:
    """Produce a few short highlights (percent change, volume spike, RSI extremes)."""
    latest = df.iloc[-1]
    last_24h = df.last('1D')
    change_24h = (latest['close'] / last_24h['close'].iloc[0] - 1) * 100 if len(last_24h) > 1 else np.nan
    avg_vol = df['volume'].rolling(window=1440, min_periods=1).mean().iloc[-1] if len(df) > 1 else df['volume'].iloc[-1]
    vol_spike = latest['volume'] > 2 * (df['volume'].rolling(window=20, min_periods=1).mean().iloc[-1])
    highlight = {
        'price': float(latest['close']),
        'change_24h_pct': float(change_24h) if not np.isnan(change_24h) else None,
        'volume': float(latest['volume']),
        'volume_spike': bool(vol_spike),
        'rsi': float(latest['rsi14'])
    }
    return highlight


def save_daily_summary(coin: str, price: float, signal: str, change_24h: float, volume: float):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('INSERT INTO daily_summary (coin, ts_utc, price, signal, change_24h, volume) VALUES (?,?,?,?,?,?)',
                (coin, datetime.now(timezone.utc).isoformat(), price, signal, change_24h, volume))
    conn.commit()
    conn.close()


# ----------------- Streamlit UI -----------------

st.set_page_config(layout='wide', page_title='Crypto Dashboard (MVP)')
st.title('Crypto Dashboard — MVP (BTC, ETH, XRP, TRX)')

init_db()

with st.sidebar:
    st.header('Opciones')
    coin_name = st.selectbox('Selecciona moneda', list(SYMBOLS.keys()), index=0)
    timeframe = st.selectbox('Timeframe', ['1m', '5m', '15m', '1h', '4h', '1d'], index=3)
    lookback = st.slider('Velas (máximo)', min_value=100, max_value=2000, value=500, step=100)
    autoreload = st.checkbox('Auto-refresh cada X segundos (usa streamlit-autorefresh)', value=True if AUTORELOAD_AVAILABLE else False)
    if AUTORELOAD_AVAILABLE and autoreload:
        interval_seconds = st.number_input('Intervalo (s)', min_value=5, max_value=600, value=15, step=5)
    else:
        interval_seconds = None
    st.markdown('---')
    st.markdown('API keys (opcionales):')
    st.markdown('- Etherscan / Deribit / TronGrid para métricas on-chain y derivados.')
    st.markdown('Si no los pones, el dashboard usa datos públicos (CoinGecko / Binance).')

symbol = SYMBOLS[coin_name]

# Autorefresh control (if enabled)
if AUTORELOAD_AVAILABLE and autoreload and interval_seconds:
    # interval is milliseconds, limit None to refresh indefinitely
    st_autorefresh(interval=interval_seconds * 1000, limit=None, key=f"autorefresh_{symbol}_{timeframe}")

# Fetch data
with st.spinner(f'Fetching {symbol} {timeframe} ...'):
    df = fetch_ohlcv_binance(symbol, timeframe=timeframe, limit=lookback)

if df is None or df.empty:
    st.error('No se pudieron obtener datos. Revisa tu conexión y vuelve a intentar.')
    st.stop()

# Compute indicators
df_ind = compute_indicators(df)
latest = df_ind.iloc[-1]
signal = generate_signal(latest)
highlights = daily_highlights(df_ind)

# Funding rate (try futures symbol like BTCUSDT)
funding = fetch_latest_funding_rate_binance(symbol.replace('/', ''))

# Layout: two columns for charts and metrics
col1, col2 = st.columns([3,1])

with col1:
    st.subheader(f'{coin_name} — {symbol} — {timeframe}')
    # Plot candlestick with plotly
    fig = go.Figure(data=[go.Candlestick(
        x=df_ind.index,
        open=df_ind['open'],
        high=df_ind['high'],
        low=df_ind['low'],
        close=df_ind['close'],
        name='Candles'
    )])

    # Add SMA/EMA lines
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['sma20'], mode='lines', name='SMA20'))
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['ema21'], mode='lines', name='EMA21'))
    fig.update_layout(height=600, margin={'t':30, 'b':10})
    st.plotly_chart(fig, use_container_width=True)

    # Volume + OBV
    st.line_chart(df_ind[['volume', 'obv']].tail(200))

with col2:
    st.metric('Último precio', f"{latest['close']:.6f}")
    st.metric('RSI (14)', f"{latest['rsi14']:.2f}")
    st.metric('MACD hist', f"{latest['macd_hist']:.6f}")
    if funding is not None:
        st.metric('Funding rate (Binance futures)', f"{funding:.6f}")
    else:
        st.text('Funding rate: no disponible (sin clave o símbolo no derivado)')

    st.markdown('### Señal compuesta')
    st.info(signal)

    st.markdown('### Highlights (última vela)')
    st.write(highlights)

    st.markdown('---')
    st.markdown('### Guardar resumen diario')
    if st.button('Guardar resumen ahora'):
        save_daily_summary(coin_name, float(latest['close']), signal, float(highlights['change_24h_pct'] or 0), float(highlights['volume']))
        st.success('Resumen guardado en la DB local.')

# Full indicator table (collapsible)
with st.expander('Tabla indicadores (últimas 50 filas)'):
    st.dataframe(df_ind.tail(50))

# Recommendations / how to use
st.markdown('---')
st.header('Recomendaciones operativas (reglas simples — educativas)')
st.markdown('''
- BUY si: precio > EMA21, EMA21 > EMA50, MACD_hist > 0, RSI < 70 y volumen por encima de la media.
- SELL si: precio < EMA21, EMA21 < EMA50 y MACD_hist < 0.
- CAUTION si: funding rate muy positiva (muchos largos apalancados) o RSI > 70.

Estas reglas son heurísticas de ejemplo — siempre gestiona el riesgo y usa stop-loss.
''')

st.markdown('---')
st.header('Siguientes mejoras (si quieres que lo continúe)')
st.markdown('''
1. Añadir websockets para updates en tiempo real (Binance, Deribit). 
2. Guardar todo en una base de datos de series temporales (InfluxDB / TimescaleDB) y usar Grafana para alertas.
3. Añadir métricas on-chain (Glassnode/CoinMetrics/Etherscan) y flujo neto a exchanges (exchange inflows/outflows).
4. Añadir análisis de derivados: open interest, funding rates históricas, skew e IV (Deribit).
5. Backtesting simple de las reglas con datos históricos.

Si quieres, te despliego el MVP en Streamlit Cloud o en un servidor con Docker.
''')

st.markdown('---')
st.write('¿Qué prefieres ahora? 1) Ver el código fuente (está en este archivo), 2) Que lo despleguemos en Streamlit Cloud y te doy los pasos exactos, 3) Añadir on-chain + derivatives en el mismo proyecto.')

# End of app

