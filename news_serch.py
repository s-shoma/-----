import streamlit as st
import feedparser
from newspaper import Article # 追加：記事を解析するライブラリ
from newspaper import Config

st.set_page_config(page_title="Myニュースキュレーター", layout="wide")
st.title("自分専用ニュース収集アプリ 📰")

st.sidebar.header("興味の設定")
keyword = st.sidebar.text_input("気になるキーワード", "半導体")

if st.sidebar.button("記事を探す"):
    rss_url = f"https://news.google.com/rss/search?q={keyword}&hl=ja&gl=JP&ceid=JP:ja"
    feed = feedparser.parse(rss_url)
    
    st.subheader(f"「{keyword}」のニュース ({len(feed.entries)}件)")
    
    if len(feed.entries) == 0:
        st.warning("記事なし")
    else:
        # プログレスバー（進行状況）を表示するとカッコいい
        progress_text = "記事を収集中..."
        my_bar = st.progress(0, text=progress_text)

        for i, entry in enumerate(feed.entries[:5]):
            my_bar.progress((i + 1) / 5, text=progress_text)

            with st.container():
                st.markdown(f"### {entry.title}")
                
                with st.expander("記事の本文をチラ見する（解析）"):
                    try:
                        # === 修正ポイント：変装設定を作る ===
                        config = Config()
                        # Chromeブラウザのふりをする設定
                        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                        
                        # 記事を取得
                        article = Article(entry.link, config=config)
                        article.download()
                        article.parse()
                        # ==================================
                        
                        if article.text:
                            st.info("▼ 抽出された本文")
                            st.write(article.text[:500] + "...") 
                            st.caption(f"[元の記事で全文を読む]({entry.link})")
                        else:
                            # うまく取れない場合はURLを表示してデバッグしやすくする
                            st.warning(f"本文が空でした。画像メインか、ブロックされています。\nURL: {entry.link}")
                            
                    except Exception as e:
                        st.error(f"読み込みエラー: {e}")
                
                st.write("---")
        
        # 完了したらプログレスバーを消す
        my_bar.empty()