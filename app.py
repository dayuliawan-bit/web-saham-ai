import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly

# Konfigurasi Halaman Web
st.set_page_config(page_title="Stock Screener & AI Predictor", layout="wide")

st.title("📈 AI Stock Screener & ML Predictor")
st.write("Cari saham undervalue, analisis grafiknya, dan prediksi harga 30 hari ke depan menggunakan Machine Learning.")

# --- FUNGSI INDIKATOR TEKNIKAL (PANDAS MURNI) ---
def calculate_indicators(df):
    # 1. Hitung RSI 14 Hari
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 2. Hitung MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Hitung SMA 200
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    return df

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("⚙️ Parameter Screener")
ticker_input = st.sidebar.text_input(
    "Daftar Saham (Pisahkan koma)", 
    "BBCA.JK, ADRO.JK, ASII.JK, BMRI.JK, ITMG.JK, PTBA.JK"
)
max_per = st.sidebar.number_input("Maksimal PER (x)", value=15.0, step=1.0)
max_pbv = st.sidebar.number_input("Maksimal PBV (x)", value=1.5, step=0.1)

# --- LOGIKA UTAMA SCREENER ---
if st.sidebar.button("🔍 Mulai Screening"):
    tickers = [t.strip().upper() for t in ticker_input.split(',')]
    progress_bar = st.progress(0)
    results = []
    
    st.session_state['stock_data'] = {} 
    
    for i, ticker in enumerate(tickers):
        with st.spinner(f"Menganalisa {ticker}..."):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                per = info.get('trailingPE', 999)
                pbv = info.get('priceToBook', 999)
                
                if per < max_per and pbv < max_pbv:
                    df = yf.download(ticker, period="2y", interval="1d", progress=False)
                    
                    if not df.empty and len(df) > 200:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                            
                        # Kalkulasi Indikator Manual
                        df = calculate_indicators(df)
                        df.dropna(inplace=True)
                        
                        st.session_state['stock_data'][ticker] = df
                        
                        latest = df.iloc[-1]
                        close_price = float(latest['Close'])
                        rsi = float(latest['RSI_14'])
                        macd_line = float(latest['MACD'])
                        signal_line = float(latest['MACD_Signal'])
                        ma200 = float(latest['SMA_200'])
                        
                        trend = "UPTREND" if close_price > ma200 else "DOWNTREND"
                        
                        if trend == "UPTREND" and macd_line > signal_line and rsi < 60:
                            signal = "🟢 BUY"
                        else:
                            signal = "⏳ WAIT & SEE"
                            
                        results.append({
                            "Saham": ticker.replace(".JK", ""),
                            "Sinyal": signal,
                            "Harga (Rp)": f"{close_price:,.0f}",
                            "PER (x)": round(per, 2),
                            "PBV (x)": round(pbv, 2),
                            "RSI": round(rsi, 2),
                            "Tren Mayor": trend
                        })
            except Exception as e:
                pass
                
        progress_bar.progress((i + 1) / len(tickers))
    
    st.session_state['results'] = results

# --- MENAMPILKAN UI (3 TAB) ---
if 'results' in st.session_state and st.session_state['results']:
    tab1, tab2, tab3 = st.tabs(["📋 Tabel Screener", "📊 Grafik Interaktif", "🤖 Prediksi AI (30 Hari)"])
    
    with tab1:
        st.subheader("Hasil Filter Valuasi Murah")
        st.dataframe(pd.DataFrame(st.session_state['results']), use_container_width=True)
        
    with tab2:
        st.subheader("Analisis Teknikal Lanjutan")
        available_tickers = list(st.session_state['stock_data'].keys())
        selected_ticker_chart = st.selectbox("Pilih Saham untuk Grafik:", available_tickers, key="chart_select")
        
        if selected_ticker_chart:
            df_plot = st.session_state['stock_data'][selected_ticker_chart]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Harga'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA_200'], line=dict(color='orange', width=2), name='MA 200'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI_14'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
            
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

            fig.update_layout(title=f"Pergerakan Harga & RSI - {selected_ticker_chart}", height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
    with tab3:
        st.subheader("Prediksi Pergerakan Harga dengan AI (Prophet)")
        st.warning("⚠️ Disclaimer: Prediksi Machine Learning didasarkan pada probabilitas tren historis dan BUKAN jaminan harga masa depan.")
        
        selected_ticker_ml = st.selectbox("Pilih Saham untuk Diprediksi:", available_tickers, key="ml_select")
        
        if st.button(f"Jalankan Prediksi 30 Hari untuk {selected_ticker_ml}"):
            with st.spinner("Model AI sedang mempelajari pola historis..."):
                df_ml = st.session_state['stock_data'][selected_ticker_ml].copy()
                df_prophet = df_ml.reset_index()[['Date', 'Close']]
                df_prophet.columns = ['ds', 'y']
                df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
                
                model = Prophet(daily_seasonality=False, yearly_seasonality=True)
                model.fit(df_prophet)
                
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                last_prediction = forecast.iloc[-1]
                current_price = df_prophet['y'].iloc[-1]
                predicted_price = last_prediction['yhat']
                trend_direction = "NAIK 🟢" if predicted_price > current_price else "TURUN 🔴"
                
                st.success(f"Prediksi Selesai! Estimasi tren 30 hari ke depan: **{trend_direction}** (Target AI: Rp{predicted_price:,.0f})")
                
                fig_prophet = plot_plotly(model, forecast)
                fig_prophet.update_layout(title=f"Forecast Harga 30 Hari - {selected_ticker_ml}", yaxis_title="Harga (Rp)", xaxis_title="Tanggal", template="plotly_dark", height=500)
                st.plotly_chart(fig_prophet, use_container_width=True)

elif 'results' in st.session_state and not st.session_state['results']:
    st.info("Tidak ada saham yang lolos kriteria fundamental.")
