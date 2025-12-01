import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.plot import plot_plotly

# ページ設定
st.set_page_config(page_title="AI株価予測アプリ 🤖", layout="wide")
st.title("Meta社『Prophet』による株価AI予測 🤖")

st.sidebar.header("設定")
ticker = st.sidebar.text_input("銘柄コード (例: 7203.T, AAPL)", "7203.T")
years = st.sidebar.slider("AIに学習させる過去年数", 1, 5, 2)
days_predict = st.sidebar.slider("向こう何日先を予測する？", 30, 365, 90)

if st.sidebar.button("AI予測を実行"):
    try:
        with st.spinner('AIが学習中...'):
            # --- 1. データ取得 ---
            start_date = datetime.now() - timedelta(days=years*365)
            end_date = datetime.now()
            
            df = yf.download(ticker, start=start_date, end=end_date)
            
            # 【重要修正】ここでデータの多重構造を平らに戻す！
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if len(df) == 0:
                st.error("データなし。コードを確認してください。")
            else:
                # --- 2. Prophet用にデータを整形 ---
                data = df.reset_index()
                
                # 日付カラムの名前を確認して統一する（DateだったりDatetimeだったりするため）
                date_col = 'Date' if 'Date' in data.columns else 'Datetime'
                
                # タイムゾーンを消す（Prophetのエラー回避）
                if date_col in data.columns:
                     # 日付型か確認してから処理
                    if pd.api.types.is_datetime64_any_dtype(data[date_col]):
                        data[date_col] = data[date_col].dt.tz_localize(None)
                
                # 必要な列だけ取り出してリネーム
                df_prophet = data[[date_col, 'Close']].rename(columns={date_col: 'ds', 'Close': 'y'})

                # --- 3. AIモデルの作成と学習 ---
                m = Prophet()
                m.fit(df_prophet)

                # --- 4. 未来予測 ---
                future = m.make_future_dataframe(periods=days_predict)
                forecast = m.predict(future)

                # --- 5. 結果表示 ---
                current_price = df['Close'].iloc[-1]
                st.metric(f"{ticker} の現在の株価", f"{float(current_price):.2f}")

                st.subheader(f"今後 {days_predict} 日間の予測チャート")
                fig = plot_plotly(m, forecast)
                fig.update_layout(
                    title="黒い点=実績, 青い線=予測, 水色の帯=予測の範囲",
                    xaxis_title="日付", 
                    yaxis_title="株価",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("予測の内訳（トレンド・周期性）"):
                    fig2 = m.plot_components(forecast)
                    st.pyplot(fig2)

    except Exception as e:
        st.error(f"エラー詳細: {e}")