from ast import List
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import groq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode 
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(model_provider="groq",model="llama-3.3-70b-versatile")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

def create_chat_graph(checkpointer):
    return graph.graph_builder.compile(checkpointer=checkpointer)

