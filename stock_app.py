import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
import urllib.parse

st.set_page_config(page_title="はまさんの投資アプリ 🚀", layout="wide")
st.title("AIシグナル推奨版 ⛩️")

# --- サイドバー設定 ---
st.sidebar.header("🛠 設定")

app_mode = st.sidebar.radio("モード選択", ["詳細分析 (単一銘柄)", "パフォーマンス比較 (複数銘柄)"])

interval_map = {"日足 (1日)": "1d", "週足 (1週間)": "1wk", "月足 (1ヶ月)": "1mo"}
selected_interval_label = st.sidebar.selectbox("チャートの足", options=interval_map.keys())
interval = interval_map[selected_interval_label]

# ==========================================
# テーマ辞書
# ==========================================
THEME_DICT = {
    "💎 レアアース関連": [
        "5713.T", "5711.T", "4063.T", "5706.T", "5727.T", "6976.T", "4005.T", "5019.T", "1605.T", "1890.T", "1885.T", "6269.T", "7013.T"
    ],
    "⚡ 半導体 (最強テーマ)": [
        "8035.T", "6857.T", "6146.T", "6920.T", "6723.T", "4063.T", "7735.T"
    ],
    "🤖 人工知能 (AI)": [
        "9984.T", "6701.T", "6702.T", "9613.T", "3993.T"
    ],
    "🚗 自動運転・EV": [
        "7203.T", "7267.T", "6758.T", "6902.T", "6594.T"
    ],
     "🏦 銀行・金融": [
        "8306.T", "8316.T", "8411.T", "8591.T", "8604.T"
    ]
}

# --- Excelリスト読み込み ---
@st.cache_data
def get_stock_list():
    try:
        df_jpx = pd.read_excel("./stock_list.xlsx")
        stock_list = []
        custom_stocks = [
            ("AAPL", "Apple Inc", "米国株: Apple", "🇺🇸 米国株"),
            ("NVDA", "NVIDIA Corp", "米国株: NVIDIA", "🇺🇸 米国株"),
            ("MSFT", "Microsoft Corp", "米国株: Microsoft", "🇺🇸 米国株"),
            ("TSLA", "Tesla Inc", "米国株: Tesla", "🇺🇸 米国株"),
            ("GOOGL", "Alphabet Inc", "米国株: Google", "🇺🇸 米国株"),
            ("AMZN", "Amazon.com", "米国株: Amazon", "🇺🇸 米国株"),
        ]
        for code, query, name, sector in custom_stocks:
            stock_list.append({"label": name, "code": code, "query": query, "sector": sector})

        for index, row in df_jpx.iterrows():
            code = str(row.iloc[1])
            name = str(row.iloc[2])
            sector = str(row.iloc[5])
            if sector == '-': sector = "その他・ETF"
            if code.isdigit() and len(code) == 4:
                full_code = f"{code}.T"
                stock_list.append({"label": f"{full_code}: {name}", "code": full_code, "query": name, "sector": sector})
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

# --- 銘柄フィルタリング ---
filtered_stocks = []
stock_labels = []

if stocks:
    theme_options = ["指定なし (全検索モード)"] + list(THEME_DICT.keys())
    selected_theme = st.sidebar.selectbox("🌟 注目テーマで絞り込み", options=theme_options)
    
    if selected_theme != "指定なし (全検索モード)":
        target_codes = THEME_DICT[selected_theme]
        filtered_stocks = [s for s in stocks if s["code"] in target_codes]
        st.sidebar.success(f"{selected_theme}: {len(filtered_stocks)}銘柄")
    else:
        unique_sectors = sorted(list(set([s["sector"] for s in stocks])))
        sector_options = ["指定なし (全銘柄)"] + unique_sectors
        selected_sector = st.sidebar.selectbox("🔍 業種で絞り込み", options=sector_options)
        if selected_sector != "指定なし (全銘柄)":
            filtered_stocks = [s for s in stocks if s["sector"] == selected_sector]
        else:
            filtered_stocks = stocks
            
    stock_labels = [s["label"] for s in filtered_stocks]

# ==========================================
# 🚀 新機能: AIシグナル・スキャナー
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 お宝銘柄発掘")
if st.sidebar.button("🔍 AI推奨銘柄をスキャン"):
    if len(filtered_stocks) > 50:
        st.sidebar.warning("銘柄数が多すぎます(50件以上)。テーマか業種で絞ってください。")
    else:
        recommended_stocks = []
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # 過去1年分のデータで判定
        scan_start = datetime.now() - timedelta(days=365)
        scan_end = datetime.now()

        for i, stock in enumerate(filtered_stocks):
            status_text.text(f"分析中: {stock['query']}...")
            progress_bar.progress((i + 1) / len(filtered_stocks))
            
            try:
                # データ取得（軽量化のため期間短縮）
                df_scan = yf.download(stock["code"], start=scan_start, end=scan_end, interval="1d", progress=False)
                if isinstance(df_scan.columns, pd.MultiIndex):
                    df_scan.columns = df_scan.columns.get_level_values(0)
                
                if len(df_scan) > 25:
                    # テクニカル計算
                    df_scan['RSI'] = calculate_rsi(df_scan['Close'])
                    df_scan['SMA25'] = df_scan['Close'].rolling(window=25).mean()
                    df_scan['SMA75'] = df_scan['Close'].rolling(window=75).mean()
                    
                    latest = df_scan.iloc[-1]
                    rsi = latest['RSI']
                    price = latest['Close']
                    sma25 = latest['SMA25']
                    sma75 = latest['SMA75']
                    
                    # --- 判定ロジック ---
                    reasons = []
                    score = 0
                    
                    # 1. RSI判定 (30以下は売られすぎ=買い)
                    if rsi <= 35:
                        reasons.append(f"💎 RSIが低い ({rsi:.1f}) - お買い得！")
                        score += 3
                    elif rsi >= 70:
                        reasons.append(f"🔥 RSIが高い ({rsi:.1f}) - 加熱気味")
                        score -= 2
                        
                    # 2. 移動平均線判定 (ゴールデンクロス風)
                    if sma25 > sma75:
                        score += 1 # 上昇トレンド中
                    
                    # スコアが高いものだけ採用
                    if score >= 1:
                        recommended_stocks.append({
                            "銘柄": stock['query'],
                            "コード": stock['code'],
                            "現在値": f"{price:.0f}円",
                            "判定": "買い推奨 🎯" if score >= 3 else "注目株 👀",
                            "理由": ", ".join(reasons) if reasons else "上昇トレンド継続中 📈"
                        })
                        
            except Exception:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # 結果表示
        if recommended_stocks:
            st.success(f"AI分析の結果、{len(recommended_stocks)}件の注目銘柄が見つかりました！")
            result_df = pd.DataFrame(recommended_stocks)
            st.table(result_df)
        else:
            st.info("現在、明確な「買いシグナル」が出ている銘柄はありませんでした。")

# ==========================================
# 🅰️ 詳細分析モード
# ==========================================
if app_mode == "詳細分析 (単一銘柄)":
    
    if not filtered_stocks:
        st.error("銘柄データが見つかりません。")
    else:
        selected_label = st.sidebar.selectbox("銘柄を選択", options=stock_labels)
        selected_data = next(s for s in filtered_stocks if s["label"] == selected_label)
        ticker = selected_data["code"]
        search_query = selected_data["query"]
        sector_name = selected_data["sector"]

        years = st.sidebar.slider("学習期間(年)", 1, 10, 5) 
        days_predict = st.sidebar.slider("予測期間(日)", 30, 365, 90)

        if st.sidebar.button("神分析を実行 ⚡"):
            try:
                with st.spinner(f'【{search_query}】を分析中...'):
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
                        badge_html = f"<span style='background-color:#333; padding:5px; border-radius:5px; font-size:14px;'>{sector_name}</span>"
                        if selected_theme != "指定なし (全検索モード)":
                             badge_html += f" <span style='background-color:#AB63FA; padding:5px; border-radius:5px; font-size:14px;'>{selected_theme}</span>"

                        st.markdown(f"## 🏢 {long_name} {badge_html}", unsafe_allow_html=True)
                        
                        pe = info.get('trailingPE', '-')
                        pb = info.get('priceToBook', '-')
                        div = info.get('dividendYield', '-')
                        if isinstance(div, (int, float)): div = f"{div*100:.2f}%"

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("現在株価", f"{float(current_price):.2f}")
                        c2.metric("PER", pe)
                        c3.metric("PBR", pb)
                        c4.metric("配当利回り", div)
                        
                        st.markdown("##### 📊 本日の詳細データ")
                        latest_row = df.iloc[-1]
                        d1, d2, d3, d4 = st.columns(4)
                        d1.metric("始値", f"{float(latest_row['Open']):.2f}")
                        d2.metric("高値", f"{float(latest_row['High']):.2f}")
                        d3.metric("安値", f"{float(latest_row['Low']):.2f}")
                        d4.metric("終値", f"{float(latest_row['Close']):.2f}")
                        
                        csv_data = convert_df_to_csv(df)
                        st.download_button(label="📥 株価データをCSVでダウンロード", data=csv_data, file_name=f"{ticker}_data.csv", mime='text/csv')
                        st.markdown("---")

                        tab1, tab2, tab3 = st.tabs(["📈 実績チャート(Pro)", "💰 決算推移", "🤖 AI予測(Pro)"])
                        no_weekends = dict(bounds=["sat", "mon"]) 

                        with tab1:
                            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                            fig.add_trace(go.Candlestick(
                                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='株価',
                                hovertemplate="<b>日付</b>: %{x|%Y/%m/%d}<br><b>始値</b>: %{open}<br><b>高値</b>: %{high}<br><b>安値</b>: %{low}<br><b>終値</b>: %{close}<extra></extra>"
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA25'], mode='lines', name='25MA', line=dict(color='#FFA500', width=1.5)), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA75'], mode='lines', name='75MA', line=dict(color='#00BFFF', width=1.5)), row=1, col=1)
                            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='出来高', marker_color='rgba(200, 200, 200, 0.5)'), row=2, col=1)
                            fig.update_layout(
                                title=f"{selected_interval_label}チャート (出来高付き)", height=600, template="plotly_dark", 
                                xaxis_rangeslider_visible=False, showlegend=True,
                                xaxis=dict(rangebreaks=[no_weekends])
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
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
                            fig_ai = plot_plotly(m, forecast)
                            fig_ai.update_layout(
                                title="AI予測信頼区間 (5年学習)", height=600, template="plotly_dark",
                                xaxis=dict(rangebreaks=[no_weekends])
                            )
                            st.plotly_chart(fig_ai, use_container_width=True)

                        st.markdown("### 📊 RSI（過熱感）")
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#AB63FA', width=2)))
                        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.2, line_width=0, annotation_text="売りゾーン", annotation_position="top left")
                        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="blue", opacity=0.2, line_width=0, annotation_text="買いゾーン", annotation_position="bottom left")
                        fig_rsi.update_layout(
                            height=300, yaxis_range=[0, 100], template="plotly_dark", title="RSI推移",
                            xaxis=dict(rangebreaks=[no_weekends])
                        )
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
    if not filtered_stocks:
        st.error("銘柄リスト読み込みエラー")
    else:
        selected_labels = st.multiselect("比較したい銘柄を選んでください", options=stock_labels, default=stock_labels[:3] if len(stock_labels)>3 else stock_labels)
        compare_years = st.sidebar.slider("比較期間(年)", 1, 10, 1)

        if st.button("比較スタート 🏁"):
            if not selected_labels:
                st.warning("銘柄を選択してください")
            else:
                try:
                    with st.spinner('データ収集中...'):
                        start_date = datetime.now() - timedelta(days=compare_years*365)
                        end_date = datetime.now()
                        results = [] 
                        combined_df = pd.DataFrame()
                        
                        for label in selected_labels:
                            target = next(s for s in filtered_stocks if s["label"] == label)
                            code = target["code"]
                            name = target["query"]
                            df = yf.download(code, start=start_date, end=end_date, interval=interval)
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            if len(df) > 0:
                                results.append({"name": name, "df": df})
                                combined_df[name] = df['Close']

                        tab_growth, tab_price = st.tabs(["📈 成長率 (%)", "💴 株価 (円)"])
                        no_weekends = dict(bounds=["sat", "mon"])

                        with tab_growth:
                            fig_comp = go.Figure()
                            for item in results:
                                df = item["df"]
                                name = item["name"]
                                initial_price = df['Close'].iloc[0]
                                df['Return'] = ((df['Close'] / initial_price) - 1) * 100
                                fig_comp.add_trace(go.Scatter(x=df.index, y=df['Return'], mode='lines', name=name))
                            
                            fig_comp.update_layout(
                                title=f"成長率比較 (%)", height=600, hovermode="x unified", template="plotly_dark",
                                xaxis=dict(rangebreaks=[no_weekends])
                            )
                            fig_comp.add_hline(y=0, line_dash="dash", line_color="gray")
                            st.plotly_chart(fig_comp, use_container_width=True)

                        with tab_price:
                            fig_price = go.Figure()
                            for item in results:
                                df = item["df"]
                                name = item["name"]
                                fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=name))
                            
                            fig_price.update_layout(
                                title=f"株価推移 (円)", height=600, hovermode="x unified", template="plotly_dark",
                                xaxis=dict(rangebreaks=[no_weekends])
                            )
                            st.plotly_chart(fig_price, use_container_width=True)

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