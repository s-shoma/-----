import streamlit as st
from datetime import datetime

st.set_page_config(page_title="はまさんの初アプリ", page_icon="🚀", layout="centered")

st.title("はまさんの初アプリ 🚀")
st.write("シンプルなあいさつアプリです。名前を入力して「送信」を押してください。")

# サイドバーで設定（言語選択など）
lang = st.sidebar.selectbox("表示言語 / Language", ("日本語", "English"))

# テキスト入力欄（セッションで入力を保持）
if "name" not in st.session_state:
    st.session_state.name = ""

name = st.text_input("あなたのお名前は？" if lang == "日本語" else "What's your name?", value=st.session_state.name, max_chars=50)

# 送信ボタン
if st.button("送信" if lang == "日本語" else "Submit"):
    st.session_state.name = name.strip()
    if not st.session_state.name:
        st.warning("名前を入力してください。" if lang == "日本語" else "Please enter your name.")
    else:
        # 時間帯であいさつを変える
        hour = datetime.now().hour
        if hour < 12:
            greet = "おはようございます" if lang == "日本語" else "Good morning"
        elif hour < 18:
            greet = "こんにちは" if lang == "日本語" else "Good afternoon"
        else:
            greet = "こんばんは" if lang == "日本語" else "Good evening"

        st.success(f"{greet}、{st.session_state.name} さん！" if lang == "日本語" else f"{greet}, {st.session_state.name}!")
        st.balloons()

# リセットボタン
if st.button("リセット" if lang == "日本語" else "Reset"):
    st.session_state.name = ""
    st.experimental_rerun()