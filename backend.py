from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import sqlite3

import os


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm, temperature=0.3)

class State(MessagesState):
    pass

def chat_node(state: State) -> State:
    messages = state["messages"]
    response = model.invoke(messages)
    return {
        "messages": [response]
    }

conn = sqlite3.connect('chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = StateGraph(State)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer = checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)