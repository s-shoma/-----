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
st.title("God Mode: 全銘柄対応 & 比較分析版 ⛩️")

# --- サイドバー設定 ---
st.sidebar.header("🛠 設定")

# 1. モード選択（ここで機能を切り替える！）
app_mode = st.sidebar.radio("モード選択", ["詳細分析 (単一銘柄)", "パフォーマンス比較 (複数銘柄)"])

# 2. 足の種類（日足・週足・月足）
interval_map = {"日足 (1日)": "1d", "週足 (1週間)": "1wk", "月足 (1ヶ月)": "1mo"}
selected_interval_label = st.sidebar.selectbox("チャートの足", options=interval_map.keys())
interval = interval_map[selected_interval_label]

# --- 関数: Excelリスト読み込み ---
@st.cache_data
def get_stock_list():
    try:
        df_jpx = pd.read_excel("./stock_list.xlsx")
        stock_list = []
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

        for index, row in df_jpx.iterrows():
            code = str(row.iloc[1])
            name = str(row.iloc[2])
            if code.isdigit() and len(code) == 4:
                full_code = f"{code}.T"
                stock_list.append({"label": f"{full_code}: {name}", "code": full_code, "query": name})
        return stock_list
    except Exception as e:
        return []

stocks = get_stock_list()

# --- 共通関数 ---
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

# ==========================================
# 🅰️ 詳細分析モード (今までの機能 + 足選択)
# ==========================================
if app_mode == "詳細分析 (単一銘柄)":
    
    if not stocks:
        st.error("銘柄リスト読み込みエラー")
    else:
        # 銘柄選択
        stock_labels = [s["label"] for s in stocks]
        selected_label = st.sidebar.selectbox("銘柄を検索・選択", options=stock_labels)
        selected_data = next(s for s in stocks if s["label"] == selected_label)
        ticker = selected_data["code"]
        search_query = selected_data["query"]

        years = st.sidebar.slider("学習期間(年)", 1, 5, 2)
        days_predict = st.sidebar.slider("予測期間(日)", 30, 365, 90)

        if st.sidebar.button("神分析を実行 ⚡"):
            try:
                with st.spinner(f'【{search_query}】を分析中...'):
                    stock_info = yf.Ticker(ticker)
                    info = stock_info.info
                    
                    start_date = datetime.now() - timedelta(days=years*365)
                    end_date = datetime.now()
                    
                    # 【変更】intervalを渡して日足・週足を切り替える
                    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    if len(df) == 0:
                        st.error("データなし")
                    else:
                        # 指標計算
                        df['RSI'] = calculate_rsi(df['Close'])
                        df['SMA25'] = df['Close'].rolling(window=25).mean()
                        df['SMA75'] = df['Close'].rolling(window=75).mean()
                        latest_rsi = df['RSI'].iloc[-1]
                        current_price = df['Close'].iloc[-1]

                        # ダッシュボード
                        long_name = info.get('longName', search_query)
                        st.markdown(f"## 🏢 {long_name}")
                        
                        pe = info.get('trailingPE', '-')
                        pb = info.get('priceToBook', '-')
                        div = info.get('dividendYield', '-')
                        if isinstance(div, (int, float)): div = f"{div*100:.2f}%"

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("現在株価", f"{float(current_price):.2f}")
                        c2.metric("PER", pe)
                        c3.metric("PBR", pb)
                        c4.metric("配当利回り", div)
                        st.markdown("---")

                        # チャート
                        tab1, tab2 = st.tabs(["📈 実績チャート", "🤖 AI予測"])
                        with tab1:
                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='株価'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA25'], mode='lines', name='25MA', line=dict(color='orange')))
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA75'], mode='lines', name='75MA', line=dict(color='blue')))
                            fig.update_layout(height=500, title=f"{selected_interval_label}チャート")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            # Prophetは日足以外だと少し精度が落ちるが動くように調整
                            data = df.reset_index()
                            date_col = 'Date' if 'Date' in data.columns else 'Datetime'
                            if date_col in data.columns:
                                if pd.api.types.is_datetime64_any_dtype(data[date_col]):
                                    data[date_col] = data[date_col].dt.tz_localize(None)
                            df_p = data[[date_col, 'Close']].rename(columns={date_col: 'ds', 'Close': 'y'})
                            m = Prophet()
                            m.fit(df_p)
                            future = m.make_future_dataframe(periods=days_predict)
                            forecast = m.predict(future)
                            fig_ai = plot_plotly(m, forecast)
                            st.plotly_chart(fig_ai, use_container_width=True)

                        # ニュース
                        st.markdown(f"### 📰 ニュース")
                        news = get_news(search_query)
                        if news:
                            for n in news:
                                with st.expander(n.title):
                                    st.markdown(f"[記事を読む]({n.link})")
                        else:
                            st.info("ニュースなし")

            except Exception as e:
                st.error(f"エラー: {e}")

# ==========================================
# 🅱️ パフォーマンス比較モード (新機能！)
# ==========================================
else:
    st.header("⚖️ 銘柄パフォーマンス比較")
    st.info("複数の銘柄を選んで、どれが一番成長したか競争させます。（スタート地点を0%として比較）")

    if not stocks:
        st.error("リスト読み込みエラー")
    else:
        # 複数選択ボックス (multiselect)
        stock_labels = [s["label"] for s in stocks]
        selected_labels = st.multiselect("比較したい銘柄を選んでください（複数可）", options=stock_labels, default=stock_labels[:2])
        
        compare_years = st.sidebar.slider("比較期間(年)", 1, 10, 1)

        if st.button("比較スタート 🏁"):
            if not selected_labels:
                st.warning("銘柄を少なくとも1つ選んでください")
            else:
                try:
                    with st.spinner('各社のデータを集めて競争させています...'):
                        start_date = datetime.now() - timedelta(days=compare_years*365)
                        end_date = datetime.now()
                        
                        fig_comp = go.Figure()
                        
                        for label in selected_labels:
                            # データ辞書からコードを取り出す
                            target = next(s for s in stocks if s["label"] == label)
                            code = target["code"]
                            
                            # データ取得
                            df = yf.download(code, start=start_date, end=end_date, interval=interval)
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            
                            if len(df) > 0:
                                # 【重要】リターン（％）に変換して比較する
                                # (今の価格 / スタート時の価格) - 1
                                # これをやらないと、1000円の株と3万円の株が比較できない！
                                initial_price = df['Close'].iloc[0]
                                df['Return'] = ((df['Close'] / initial_price) - 1) * 100
                                
                                fig_comp.add_trace(go.Scatter(
                                    x=df.index, 
                                    y=df['Return'], 
                                    mode='lines', 
                                    name=f"{target['query']} ({code})"
                                ))

                        fig_comp.update_layout(
                            title=f"過去{compare_years}年間の成長率比較 (%)",
                            xaxis_title="日付",
                            yaxis_title="リターン (%)",
                            height=600,
                            hovermode="x unified" # カーソルを合わせた時に全銘柄の数値を表示
                        )
                        # 0%のラインを引く
                        fig_comp.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"比較エラー: {e}")