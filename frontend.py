import streamlit as st
from backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

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

    response = chatbot.invoke({'messages': [user_message]}, config=CONFIG)
    ai_message = response["messages"][-1].content
    st.session_state["messages"].append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)