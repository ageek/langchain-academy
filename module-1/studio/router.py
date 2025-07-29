import os 
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_sambanova import ChatSambaNovaCloud
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM with bound tool
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

llm_with_tools = my_llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()