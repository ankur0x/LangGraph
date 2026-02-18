import os
import requests
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode 

load_dotenv()

# 1. STATE 
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. TOOLS

@tool
def get_weather(city: str) -> str:
    """Takes a city name as an input and returns the current weather for the city."""
    print(f"🔨 Tool Called: get_weather({city})")
    url = f"http://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"

@tool
def run_command(command: str) -> str:
    """Takes a shell command as input to execute on system and returns output."""
    print(f"🔨 Tool Called: run_command({command})")
    result = os.popen(command).read()
    return result if result else "Command executed with no output."

tools = [get_weather, run_command]

# 3. LLM 
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful AI Assistant who resolves user queries.
You have access to tools. Use them step by step when needed.
Always reason through the problem before calling a tool."""

# 4. NODES

def call_llm(state: AgentState) -> AgentState:
    """Node: invoke the LLM, potentially requesting tool calls."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 5. ROUTER (conditional edge)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"

# 6. BUILD THE GRAPH 

graph_builder = StateGraph(AgentState)

# Register nodes
graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)

# Entry point
graph_builder.set_entry_point("llm")

# Conditional edge: after LLM runs, decide where to go
graph_builder.add_conditional_edges(
    "llm",                    
    should_continue,          
    {
        "tools": "tools",    
        "end": END           
    }
)

# After tools run, ALWAYS go back to LLM (observe → re-plan loop)
graph_builder.add_edge("tools", "llm")

# Compile into a runnable
agent = graph_builder.compile()

# 7. CONVERSATION LOOP 

print("🤖 LangGraph Agent Ready (type 'exit' to quit)\n")

# Persist messages across turns for multi-turn memory
conversation_history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    if not user_input:
        continue

    # Add the new human message to history
    conversation_history.append(HumanMessage(content=user_input))

    # Invoke the graph with full history
    result = agent.invoke({"messages": conversation_history})

    # Extract the final AI response (last message)
    final_message = result["messages"][-1]
    print(f"\n🤖: {final_message.content}\n")

    # Update history with all new messages (tool calls, results, final answer)
    conversation_history = result["messages"]
