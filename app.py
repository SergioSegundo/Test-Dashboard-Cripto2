import requests
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta, timezone

# Mapeo de símbolos para CoinGecko
COINGECKO_IDS = {
    'Bitcoin': 'bitcoin',
    'Ethereum': 'ethereum',
    'XRP': 'ripple',
    'Tron': 'tron'
}

@st.cache_data(ttl=600)
def fetch_ohlcv_coingecko(coin_name: str, vs_currency='usd', days=30, interval='daily'):
    """
    Obtiene OHLCV desde CoinGecko para un periodo dado.
    - coin_name: 'Bitcoin', 'Ethereum', etc.
    - vs_currency: moneda contra la que cotiza, por defecto usd
    - days: últimos días para obtener (máximo 90 para hourly)
    - interval: 'daily' o 'hourly'
    
    Devuelve un DataFrame con timestamp, open, high, low, close, volume
    """
    coin_id = COINGECKO_IDS.get(coin_name)
    if coin_id is None:
        st.error(f"Moneda {coin_name} no soportada en CoinGecko.")
        return None
    
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': interval
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # 'prices', 'market_caps', 'total_volumes' están en listas [timestamp, value]
        prices = data['prices']  # [[ts, price], ...]
        volumes = data['total_volumes']  # [[ts, volume], ...]
        
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
        df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
        
        df = pd.merge(df_prices, df_volumes, on='timestamp')
        
        # Como no nos dan OHLC directo, approximamos con precio (close), open, high, low = price (no tenemos detalles intravelas)
        df['open'] = df['price']
        df['high'] = df['price']
        df['low'] = df['price']
        df['close'] = df['price']
        
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        st.error(f"Error obteniendo datos de CoinGecko: {e}")
        return None
