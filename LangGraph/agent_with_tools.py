from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def search_wiki(query: str):
    """Searches DuckDuckGo using LangChain's DuckDuckGoSearchRun tool."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    return wiki_tool.invoke(query)

# Step 1: Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Load environment variables
load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

tools = [search_wiki]

llm_with_tools = llm.bind_tools(tools)

# Define chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


from langgraph.prebuilt import ToolNode, tools_condition

graph_builder = StateGraph(State)

# Define nodes
graph_builder.add_node("assistant",chatbot)
graph_builder.add_node("tools",ToolNode(tools))

#define edges
graph_builder.add_edge(START,"assistant")
graph_builder.add_conditional_edges("assistant",tools_condition)
graph_builder.add_edge("tools","assistant")


# Compile the graph
graph = graph_builder.compile()


user_msg="what is quantum computing?"
response = graph.invoke({"messages": [HumanMessage(content=user_msg )]})


for m in response["messages"]:
    m.pretty_print()