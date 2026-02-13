from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
import sqlite3

import os


load_dotenv()

# MODEL
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"   
)
model = ChatHuggingFace(llm=llm, temperature=0.3)

# STATE
class State(MessagesState):
    pass

# TOOLS
search_tool = DuckDuckGoSearchResults(output_format="string")

@tool
def calculator_tool(number1: float, number2: float, operation: str) -> float:
    """
    Performs a basic arithmetic operation on two numbers.
    Supported Operations: add, subtract, multiply, divide
    """
    if operation == "add":
        return number1 + number2
    elif operation == "subtract":
        return number1 - number2
    elif operation == "multiply":
        return number1 * number2
    elif operation == "divide":
        if number2 == 0:
            raise ValueError("Cannot divide by zero.")
        return number1 / number2
    else:
        raise ValueError(f"Unsupported operation: {operation}")

# NODES
def chat_node(state: State) -> State: # node 1
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {
        "messages": [response]
    }

tools  = [search_tool, calculator_tool]
model_with_tools = model.bind_tools(tools)

tool_node = ToolNode(tools) # node 2

# SQLITE
conn = sqlite3.connect('chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Graph
graph = StateGraph(State)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer = checkpointer)

# HELPER FUNCTION TO RETRIEVE ALL THREADS
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)