from typing import List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# Step 1: Define State
class State(Dict):
    messages: List[Dict[str, str]] 
    
# Load environment variables
load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)



# Define chatbot function
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    state["messages"].append({"role": "assistant", "content": response})  # Treat response as a string
    return {"messages": state["messages"]}


# Step 2: Initialize StateGraph
graph_builder = StateGraph(State)

# Add nodes and edges
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)


# Compile the graph
graph = graph_builder.compile()


user_msg="what is quantum computing?"
result = graph.invoke({"messages": [user_msg]})
print(' \n ')
print('User : ', user_msg )
print('Agent : ',result["messages"][-1]["content"].content)

