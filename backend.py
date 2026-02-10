from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

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

checkpointer = MemorySaver()
graph = StateGraph(State)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer = checkpointer)