import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import feedparser
import urllib.parse

st.set_page_config(page_title="はまさんの神投資アプリ 🚀", layout="wide")
st.title("God Mode: 全銘柄対応版 ⛩️")

st.sidebar.header("設定")

# --- 関数: 東証のExcelリストを読み込む ---
@st.cache_data
def get_stock_list():
    try:
        # Excelファイルを読み込む（ヘッダーなどの調整）
        df_jpx = pd.read_excel("./stock_list.xlsx")
        
        # 必要な列だけ残す（コードと銘柄名）
        # データによっては列名が異なる場合があるので注意
        # 一般的に2列目がコード、3列目が銘柄名
        stock_list = []
        
        # 米国株などの人気銘柄を手動で先頭に追加
        custom_stocks = [
            ("AAPL", "Apple Inc", "米国株: Apple"),
            ("NVDA", "NVIDIA Corp", "米国株: NVIDIA"),
            ("MSFT", "Microsoft Corp", "米国株: Microsoft"),
            ("TSLA", "Tesla Inc", "米国株: Tesla"),
            ("GOOGL", "Alphabet Inc", "米国株: Google"),
            ("AMZN", "Amazon.com", "米国株: Amazon"),
        ]
        
        for code, query, name in custom_stocks:
            stock_list.append({"label": name, "code": code, "query": query})

        # Excelのデータを追加
        for index, row in df_jpx.iterrows():
            code = str(row.iloc[1]) # コード
            name = str(row.iloc[2]) # 銘柄名
            
            # 4桁の数字コードのみを対象にする
            if code.isdigit() and len(code) == 4:
                full_code = f"{code}.T" # yfinance用に .T をつける
                stock_list.append({
                    "label": f"{full_code}: {name}", # 表示名
                    "code": full_code,               # 実際のコード
                    "query": name                    # ニュース検索用
                })
                
        return stock_list
    except Exception as e:
        st.error(f"Excel読み込みエラー: {e}")
        return []

# --- 銘柄選択エリア ---
stocks = get_stock_list()

if not stocks:
    st.warning("stock_list.xls が見つかりません。プロジェクトフォルダに配置してください。")
    # エラー時のフォールバック
    ticker = "7203.T"
    search_query = "トヨタ自動車"
else:
    # 選択ボックスを作成（表示名を使う）
    stock_labels = [s["label"] for s in stocks]
    selected_label = st.sidebar.selectbox("銘柄を検索・選択", options=stock_labels)
    
    # 選ばれたラベルから、コードと検索ワードを取り出す
    selected_data = next(s for s in stocks if s["label"] == selected_label)
    ticker = selected_data["code"]
    search_query = selected_data["query"]
    
    st.sidebar.write(f"選択中: {ticker}")

years = st.sidebar.slider("学習期間(年)", 1, 5, 2)
days_predict = st.sidebar.slider("予測期間(日)", 30, 365, 90)

# --- 関数群 ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_news(query):
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(rss_url)
    return feed.entries[:5]

if st.sidebar.button("神分析を実行 ⚡"):
    try:
        with st.spinner(f'【{search_query}】のデータを収集中...'):
            # ファンダメンタルズ
            stock_info = yf.Ticker(ticker)
            info = stock_info.info
            
            start_date = datetime.now() - timedelta(days=years*365)
            end_date = datetime.now()
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) == 0:
                st.error("データなし")
            else:
                df['RSI'] = calculate_rsi(df['Close'])
                df['SMA25'] = df['Close'].rolling(window=25).mean()
                df['SMA75'] = df['Close'].rolling(window=75).mean()
                latest_rsi = df['RSI'].iloc[-1]
                current_price = df['Close'].iloc[-1]

                # ダッシュボード
                long_name = info.get('longName', search_query)
                st.markdown(f"## 🏢 {long_name}")
                
                pe_ratio = info.get('trailingPE', '-')
                pb_ratio = info.get('priceToBook', '-')
                dividend = info.get('dividendYield', 0)
                if dividend is not None and dividend != '-' and isinstance(dividend, (int, float)):
                    dividend = f"{dividend * 100:.2f}%"
                else:
                    dividend = "-"

                col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                col_f1.metric("現在の株価", f"{float(current_price):.2f}")
                col_f2.metric("PER", pe_ratio)
                col_f3.metric("PBR", pb_ratio)
                col_f4.metric("配当利回り", dividend)
                
                st.markdown("---")

                # 判定
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

                # チャート
                tab1, tab2 = st.tabs(["📈 実績チャート", "🤖 未来予測チャート"])
                with tab1:
                    fig_candle = go.Figure()
                    fig_candle.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='株価'))
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

                # ニュース
                st.markdown(f"### 📰 「{search_query}」の最新ニュース")
                news_entries = get_news(search_query)
                if news_entries:
                    for entry in news_entries:
                        published = entry.published if 'published' in entry else ""
                        with st.expander(f"{entry.title} ({published})"):
                            st.write(f"Source: {entry.source.title if 'source' in entry else 'Google'}")
                            st.markdown(f"[記事を読む]({entry.link})")
                else:
                    st.info("ニュースなし")

    except Exception as e:
        st.error(f"エラー: {e}")