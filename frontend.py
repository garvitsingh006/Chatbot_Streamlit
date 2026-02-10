import streamlit as st
from backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

# import os
# st.write("KEY EXISTS:", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_message = st.chat_input("Type here: ")

if user_message:
    st.session_state["messages"].append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_message)]},
                config = CONFIG,
                stream_mode="messages"
            )
        )
    st.session_state["messages"].append({"role": "assistant", "content": ai_message})