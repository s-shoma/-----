import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots # グラフを重ねるための道具
import feedparser
import urllib.parse

st.set_page_config(page_title="はまさんの神投資アプリ 🚀", layout="wide")
st.title("God Mode: 視覚強化版 (Visual Pro) ⛩️")

# --- サイドバー設定 ---
st.sidebar.header("🛠 設定")

app_mode = st.sidebar.radio("モード選択", ["詳細分析 (単一銘柄)", "パフォーマンス比較 (複数銘柄)"])

interval_map = {"日足 (1日)": "1d", "週足 (1週間)": "1wk", "月足 (1ヶ月)": "1mo"}
selected_interval_label = st.sidebar.selectbox("チャートの足", options=interval_map.keys())
interval = interval_map[selected_interval_label]

# --- Excelリスト読み込み ---
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

def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8-sig')

# ==========================================
# 🅰️ 詳細分析モード
# ==========================================
if app_mode == "詳細分析 (単一銘柄)":
    
    if not stocks:
        st.error("銘柄リスト読み込みエラー")
    else:
        stock_labels = [s["label"] for s in stocks]
        selected_label = st.sidebar.selectbox("銘柄を検索・選択", options=stock_labels)
        selected_data = next(s for s in stocks if s["label"] == selected_label)
        ticker = selected_data["code"]
        search_query = selected_data["query"]

        years = st.sidebar.slider("学習期間(年)", 1, 5, 2)
        days_predict = st.sidebar.slider("予測期間(日)", 30, 365, 90)

        if st.sidebar.button("神分析を実行 ⚡"):
            try:
                with st.spinner(f'【{search_query}】を詳細分析中...'):
                    stock_info = yf.Ticker(ticker)
                    info = stock_info.info
                    financials = stock_info.financials
                    
                    start_date = datetime.now() - timedelta(days=years*365)
                    end_date = datetime.now()
                    
                    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
                    
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
                        
                        csv_data = convert_df_to_csv(df)
                        st.download_button(label="📥 株価データをCSVでダウンロード", data=csv_data, file_name=f"{ticker}_data.csv", mime='text/csv')
                        st.markdown("---")

                        # チャートタブ
                        tab1, tab2, tab3 = st.tabs(["📈 実績チャート(Pro)", "💰 決算推移", "🤖 AI予測(Pro)"])
                        
                        # --- 1. 実績チャート (Visual Upgrade) ---
                        with tab1:
                            # 2段組みのグラフを作る（上が株価、下が出来高）
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                vertical_spacing=0.03, row_heights=[0.7, 0.3])

                            # ローソク足
                            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='株価'), row=1, col=1)
                            # 移動平均線
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA25'], mode='lines', name='25MA', line=dict(color='#FFA500', width=1.5)), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA75'], mode='lines', name='75MA', line=dict(color='#00BFFF', width=1.5)), row=1, col=1)
                            
                            # 出来高（棒グラフ）
                            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='出来高', marker_color='rgba(200, 200, 200, 0.5)'), row=2, col=1)

                            # デザイン調整（ダークモード風）
                            fig.update_layout(
                                title=f"{selected_interval_label}チャート (出来高付き)",
                                height=600,
                                template="plotly_dark", # ★ここが魔法の言葉！黒背景になります
                                xaxis_rangeslider_visible=False,
                                showlegend=True,
                                margin=dict(l=20, r=20, t=50, b=20)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # --- 2. 決算グラフ ---
                        with tab2:
                            if financials is not None and not financials.empty:
                                fin_df = financials.T.sort_index()
                                fig_fin = go.Figure()
                                if 'Total Revenue' in fin_df.columns:
                                    fig_fin.add_trace(go.Bar(x=fin_df.index, y=fin_df['Total Revenue'], name='売上高', marker_color='#00CC96'))
                                if 'Net Income' in fin_df.columns:
                                    fig_fin.add_trace(go.Bar(x=fin_df.index, y=fin_df['Net Income'], name='純利益', marker_color='#EF553B'))

                                fig_fin.update_layout(title="業績推移", barmode='group', height=500, template="plotly_dark")
                                st.plotly_chart(fig_fin, use_container_width=True)
                            else:
                                st.info("決算データなし")

                        # --- 3. AI予測 (Visual Upgrade) ---
                        with tab3:
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
                            
                            # Prophetの図もPlotlyでカスタマイズ
                            fig_ai = plot_plotly(m, forecast)
                            fig_ai.update_layout(
                                title="AI予測信頼区間 (黒背景版)",
                                height=600,
                                template="plotly_dark", # 黒背景
                                xaxis_title="日付", yaxis_title="株価"
                            )
                            st.plotly_chart(fig_ai, use_container_width=True)

                        # --- 4. RSI (Visual Upgrade: 色帯付き) ---
                        st.markdown("### 📊 RSI（過熱感）")
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#AB63FA', width=2)))
                        
                        # 帯（ゾーン）をつける
                        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2, line_width=0, annotation_text="売りゾーン", annotation_position="top left")
                        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="blue", opacity=0.2, line_width=0, annotation_text="買いゾーン", annotation_position="bottom left")
                        
                        fig_rsi.update_layout(height=300, yaxis_range=[0, 100], template="plotly_dark", title="RSI推移 (70以上=赤 / 30以下=青)")
                        st.plotly_chart(fig_rsi, use_container_width=True)

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
# 🅱️ パフォーマンス比較モード
# ==========================================
else:
    st.header("⚖️ 銘柄パフォーマンス比較")
    if not stocks:
        st.error("リスト読み込みエラー")
    else:
        stock_labels = [s["label"] for s in stocks]
        selected_labels = st.multiselect("比較したい銘柄を選んでください", options=stock_labels, default=stock_labels[:3])
        compare_years = st.sidebar.slider("比較期間(年)", 1, 10, 1)

        if st.button("比較スタート 🏁"):
            if not selected_labels:
                st.warning("銘柄を選択してください")
            else:
                try:
                    with st.spinner('データ収集中...'):
                        start_date = datetime.now() - timedelta(days=compare_years*365)
                        end_date = datetime.now()
                        fig_comp = go.Figure()
                        combined_df = pd.DataFrame()
                        
                        for label in selected_labels:
                            target = next(s for s in stocks if s["label"] == label)
                            code = target["code"]
                            name = target["query"]
                            df = yf.download(code, start=start_date, end=end_date, interval=interval)
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            if len(df) > 0:
                                initial_price = df['Close'].iloc[0]
                                df['Return'] = ((df['Close'] / initial_price) - 1) * 100
                                fig_comp.add_trace(go.Scatter(x=df.index, y=df['Return'], mode='lines', name=f"{name}"))
                                combined_df[name] = df['Close']

                        fig_comp.update_layout(
                            title=f"成長率比較 (%) - Dark Mode",
                            height=600, hovermode="x unified",
                            template="plotly_dark" # ここも黒背景
                        )
                        fig_comp.add_hline(y=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_comp, use_container_width=True)

                        if len(combined_df.columns) > 1:
                            csv_comp = convert_df_to_csv(combined_df)
                            st.download_button(label="データをダウンロード", data=csv_comp, file_name="comparison.csv", mime='text/csv')

                            st.markdown("### 🧩 相関ヒートマップ")
                            corr_matrix = combined_df.corr()
                            fig_heat = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                                colorscale='RdBu_r', zmin=-1, zmax=1,
                                text=corr_matrix.values, texttemplate="%{text:.2f}"
                            ))
                            fig_heat.update_layout(height=600, template="plotly_dark")
                            st.plotly_chart(fig_heat, use_container_width=True)
                        else:
                            st.info("※2つ以上選んでください")

                except Exception as e:
                    st.error(f"比較エラー: {e}")