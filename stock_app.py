import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

st.set_page_config(page_title="はまさんの最強投資アプリ 🚀", layout="wide")
st.title("AI × テクニカル分析 投資判定アプリ 💹")

st.sidebar.header("設定")

# --- 1. 銘柄辞書 ---
stock_dict = {
    "トヨタ自動車 (7203)": "7203.T",
    "ソニーグループ (6758)": "6758.T",
    "任天堂 (7974)": "7974.T",
    "ソフトバンクG (9984)": "9984.T",
    "三菱UFJ (8306)": "8306.T",
    "東京エレクトロン (8035)": "8035.T",
    "キーエンス (6861)": "6861.T",
    "ファーストリテイリング (9983)": "9983.T",
    "Apple (AAPL)": "AAPL",
    "NVIDIA (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "★その他（手動入力）": "MANUAL"
}

selected_name = st.sidebar.selectbox("銘柄を選択してください", options=stock_dict.keys())

if stock_dict[selected_name] == "MANUAL":
    ticker = st.sidebar.text_input("銘柄コードを入力 (例: 7203.T)", "7203.T")
else:
    ticker = stock_dict[selected_name]
    st.sidebar.write(f"選択中: {ticker}")

years = st.sidebar.slider("学習期間(年)", 1, 5, 2)
days_predict = st.sidebar.slider("予測期間(日)", 30, 365, 90)

# --- 関数: RSI計算 ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if st.sidebar.button("分析・予測を実行"):
    try:
        with st.spinner('AIとテクニカル指標を計算中...'):
            start_date = datetime.now() - timedelta(days=years*365)
            end_date = datetime.now()
            df = yf.download(ticker, start=start_date, end=end_date)
            
            # 多重インデックス対策
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) == 0:
                st.error("データなし")
            else:
                # --- テクニカル指標の計算 ---
                df['RSI'] = calculate_rsi(df['Close'])
                # 移動平均線（トレンドを見る線）
                df['SMA25'] = df['Close'].rolling(window=25).mean()
                df['SMA75'] = df['Close'].rolling(window=75).mean()
                
                latest_rsi = df['RSI'].iloc[-1]
                
                # --- 判定ロジック ---
                signal = "様子見 🍵"
                if latest_rsi >= 70:
                    signal = "売りシグナル（買われすぎ） 🔥"
                elif latest_rsi <= 30:
                    signal = "買いシグナル（売られすぎ） 💎"

                # --- 1. 結果サマリー ---
                st.subheader(f"銘柄: {ticker} の分析結果")
                col1, col2, col3 = st.columns(3)
                current_price = df['Close'].iloc[-1]
                
                col1.metric("現在の株価", f"{float(current_price):.2f}")
                col2.metric("RSI (過熱感)", f"{latest_rsi:.1f}", "70以上で売り/30以下で買い")
                
                if latest_rsi >= 70:
                    col3.error(f"判定: {signal}")
                elif latest_rsi <= 30:
                    col3.success(f"判定: {signal}")
                else:
                    col3.info(f"判定: {signal}")

                # --- 2. メインチャート（ローソク足）の描画 ---
                st.markdown("### 📈 株価チャート（実績）")
                
                fig_candle = go.Figure()

                # ローソク足
                fig_candle.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    name='株価'
                ))

                # 移動平均線（短期・長期）
                fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA25'], mode='lines', name='25日移動平均', line=dict(color='orange', width=1)))
                fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA75'], mode='lines', name='75日移動平均', line=dict(color='blue', width=1)))

                fig_candle.update_layout(
                    height=500,
                    xaxis_rangeslider_visible=False, # 下のスライダーを消す
                    title=f"{ticker} のローソク足チャート"
                )
                st.plotly_chart(fig_candle, use_container_width=True)

                # --- 3. Prophet AI予測 ---
                st.markdown("### 🤖 AIによる未来予測")
                
                # Prophet用データ整形
                data = df.reset_index()
                date_col = 'Date' if 'Date' in data.columns else 'Datetime'
                if date_col in data.columns:
                    if pd.api.types.is_datetime64_any_dtype(data[date_col]):
                        data[date_col] = data[date_col].dt.tz_localize(None)
                
                df_prophet = data[[date_col, 'Close']].rename(columns={date_col: 'ds', 'Close': 'y'})
                
                m = Prophet()
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=days_predict)
                forecast = m.predict(future)
                
                fig_ai = plot_plotly(m, forecast)
                fig_ai.update_layout(height=500, title="青い線がAIの予測値です")
                st.plotly_chart(fig_ai, use_container_width=True)

                # --- 4. RSIグラフ ---
                st.markdown("### 📊 RSI（過熱感）の推移")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="blue")
                fig_rsi.update_layout(height=300, yaxis_range=[0, 100])
                st.plotly_chart(fig_rsi, use_container_width=True)

    except Exception as e:
        st.error(f"エラー: {e}")