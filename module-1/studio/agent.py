import os
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

from langchain_sambanova import ChatSambaNovaCloud
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# Define LLM with bound tools

api_key = os.getenv("SAMBANOVA_API_KEY")
if not api_key:
    raise ValueError("SAMBANOVA_API_KEY environment variable is not set.")


samba_api_key = os.getenv("SAMBANOVA_API_KEY")

my_llm = ChatSambaNovaCloud(
    sambanova_api_key=samba_api_key,
    # model="Llama-4-Maverick-17B-128E-Instruct",
    model="DeepSeek-V3-0324",
    max_tokens=4096,
    temperature=0.0,
    top_p=0.01,
)

llm_with_tools = my_llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
graph = builder.compile()
