from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, BaseTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import aiosqlite
import asyncio
import threading

import os


load_dotenv()

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

# -------------------
# 1. LLM
# -------------------
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"   
)
model = ChatHuggingFace(llm=llm, temperature=0.3)


# -------------------
# 2. Tools
# -------------------
# search_tool = DuckDuckGoSearchResults(output_format="string")

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
    
client = MultiServerMCPClient(
    {
        "serpapi": {
            "transport": "streamable_http",
            "url": "https://mcp.serpapi.com/c7a976835999649b411f090682ebeacb6a1e3c3eec7926893f81d051c2350ec5/mcp"
        }
    }
)
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

mcp_tools = load_mcp_tools()
tools  = [calculator_tool, *mcp_tools]
model_with_tools = model.bind_tools(tools) if tools else model

# -------------------
# 3. State
# -------------------
class State(MessagesState):
    pass

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: State) -> State: # node 1 
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await model_with_tools.ainvoke(messages)
    return {
        "messages": [response]
    }


tool_node = ToolNode(tools) if tools else None # node 2

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(State)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def retrieve_all_threads():
    return run_async(_alist_threads())