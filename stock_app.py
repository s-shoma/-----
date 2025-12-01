import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import feedparser # ニュース収集の最強ライブラリを追加

st.set_page_config(page_title="はまさんの神投資アプリ 🚀", layout="wide")
st.title("God Mode: AI × ファンダメンタルズ × ニュース ⛩️")

st.sidebar.header("設定")

# --- 1. 銘柄辞書 ---
# 名前からコードだけでなく、ニュース検索用の「検索ワード」も取得できるように工夫します
# 形式: "表示名": ("コード", "ニュース検索ワード")
stock_dict = {
    "トヨタ自動車 (7203)": ("7203.T", "トヨタ自動車"),
    "ソニーグループ (6758)": ("6758.T", "ソニーグループ"),
    "任天堂 (7974)": ("7974.T", "任天堂"),
    "ソフトバンクG (9984)": ("9984.T", "ソフトバンクグループ"),
    "三菱UFJ (8306)": ("8306.T", "三菱UFJ"),
    "東京エレクトロン (8035)": ("8035.T", "東京エレクトロン"),
    "キーエンス (6861)": ("6861.T", "キーエンス"),
    "ファーストリテイリング (9983)": ("9983.T", "ファーストリテイリング"),
    "日立製作所 (6501)": ("6501.T", "日立製作所"),
    "Apple (AAPL)": ("AAPL", "Apple Inc"),
    "NVIDIA (NVDA)": ("NVDA", "NVIDIA"),
    "Microsoft (MSFT)": ("MSFT", "Microsoft"),
    "Google (GOOGL)": ("GOOGL", "Google Alphabet"),
    "Tesla (TSLA)": ("TSLA", "Tesla Inc"),
    "Amazon (AMZN)": ("AMZN", "Amazon.com"),
    "★その他（手動入力）": ("MANUAL", "MANUAL")
}

selected_name = st.sidebar.selectbox("銘柄を選択してください", options=stock_dict.keys())

if selected_name == "★その他（手動入力）":
    ticker = st.sidebar.text_input("銘柄コード (例: 7203.T)", "7203.T")
    search_query = st.sidebar.text_input("ニュース検索ワード (例: トヨタ)", "トヨタ")
else:
    ticker = stock_dict[selected_name][0]       # コード (例: 7203.T)
    search_query = stock_dict[selected_name][1] # 検索ワード (例: トヨタ自動車)
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

# --- 関数: Googleニュースを取得する ---
def get_news(query):
    # GoogleニュースのRSS URL (日本語設定)
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(rss_url)
    return feed.entries[:5] # 最新5件を返す

if st.sidebar.button("神分析を実行 ⚡"):
    try:
        with st.spinner('株価データと最新ニュースを集めています...'):
            # --- 1. ファンダメンタルズ ---
            stock_info = yf.Ticker(ticker)
            info = stock_info.info
            
            # データ取得
            start_date = datetime.now() - timedelta(days=years*365)
            end_date = datetime.now()
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) == 0:
                st.error("データなし")
            else:
                # --- テクニカル指標 ---
                df['RSI'] = calculate_rsi(df['Close'])
                df['SMA25'] = df['Close'].rolling(window=25).mean()
                df['SMA75'] = df['Close'].rolling(window=75).mean()
                latest_rsi = df['RSI'].iloc[-1]
                current_price = df['Close'].iloc[-1]

                # --- 2. ダッシュボード表示 ---
                long_name = info.get('longName', search_query) # 名前が取れなければ検索ワードを表示
                st.markdown(f"## 🏢 {long_name} の分析")
                
                pe_ratio = info.get('trailingPE', '-')
                pb_ratio = info.get('priceToBook', '-')
                dividend = info.get('dividendYield', 0)
                if dividend is not None and dividend != '-' and isinstance(dividend, (int, float)):
                    dividend = f"{dividend * 100:.2f}%"
                else:
                    dividend = "-"

                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                col_f1.metric("現在の株価", f"{float(current_price):.2f}")
                col_f2.metric("PER (割安度)", pe_ratio)
                col_f3.metric("PBR (資産倍率)", pb_ratio)
                col_f4.metric("配当利回り", dividend)

                st.markdown("---")

                # --- 3. 投資判断シグナル ---
                st.subheader("🤖 AI & テクニカル判定")
                col1, col2 = st.columns(2)
                
                signal = "様子見 🍵"
                if latest_rsi >= 70:
                    signal = "売りシグナル（買われすぎ） 🔥"
                    col1.error(f"RSI判定: {signal}")
                elif latest_rsi <= 30:
                    signal = "買いシグナル（売られすぎ） 💎"
                    col1.success(f"RSI判定: {signal}")
                else:
                    col1.info(f"RSI判定: {signal}")
                
                col2.metric("現在のRSI", f"{latest_rsi:.1f}")

                # --- 4. チャート表示 ---
                tab1, tab2 = st.tabs(["📈 実績チャート", "🤖 未来予測チャート"])
                
                with tab1:
                    fig_candle = go.Figure()
                    fig_candle.add_trace(go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='株価'
                    ))
                    fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA25'], mode='lines', name='25日線', line=dict(color='orange', width=1)))
                    fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA75'], mode='lines', name='75日線', line=dict(color='blue', width=1)))
                    fig_candle.update_layout(height=500, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_candle, use_container_width=True)
                
                with tab2:
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
                    fig_ai.update_layout(height=500)
                    st.plotly_chart(fig_ai, use_container_width=True)

                # --- 5. 最新ニュース（ここをGoogleニュースに変更！） ---
                st.markdown(f"### 📰 「{search_query}」の最新ニュース")
                news_entries = get_news(search_query)
                
                if news_entries:
                    for entry in news_entries:
                        # ニュースの日付を取得
                        published = entry.published if 'published' in entry else "日付不明"
                        with st.expander(f"{entry.title} ({published})"):
                            st.write(f"情報源: {entry.source.title if 'source' in entry else 'Googleニュース'}")
                            st.markdown(f"[記事を読む]({entry.link})")
                else:
                    st.info("関連ニュースが見つかりませんでした。")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")