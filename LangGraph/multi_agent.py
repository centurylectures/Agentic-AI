from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()
# Initialize OpenAI model
model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)




def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

def search_wiki(query: str):
    """Searches DuckDuckGo using LangChain's DuckDuckGoSearchRun tool."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    return wiki_tool.invoke(query)


research_agent = create_react_agent(
    model=model,
    tools=[search_wiki],
    name="research_expert",
    prompt="You are a world class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile and run
app = workflow.compile()

result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what is quantum computing?"
        }
    ]
})

for m in result["messages"]:
    m.pretty_print()


# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what is the trump born year. Multiply it by 2 and add 5"
        }
    ]
})

for m in result["messages"]:
    m.pretty_print()