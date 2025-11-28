import streamlit as st
import time

st.title("AIチャット簡易版 🤖")

# 1. 「会話の履歴」を保存する場所を作る
# Streamlitはボタンを押すたびにリセットされるので、
# "session_state" という場所に履歴を避難させておく必要があります。
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 過去のやり取りを画面に表示しなおす
# これがないと、新しい発言をするたびに過去の会話が消えてしまいます。
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 3. 入力欄を表示し、入力されたら処理スタート
if prompt := st.chat_input("何か話しかけてみて！"):
    
    # A. ユーザーの入力（prompt）を表示・保存
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- ここからAIのターン ---
    
    # B. AIの返答を作る（今はオウム返し）
    response = f"なるほど、「{prompt}」なんですね！"
    
    # C. AIの返答を表示・保存
    with st.chat_message("assistant"):
        # ちょっと考えているフリをする演出（0.5秒待つ）
        time.sleep(0.5)
        st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})