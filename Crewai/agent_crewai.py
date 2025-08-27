
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Load LLM Model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

from crewai_tools import WebsiteSearchTool
import os

# Instantiate Web Search Tool
web_search_tool = WebsiteSearchTool()

from crewai import Agent

# Create Agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of the AI industry',
    backstory='An expert analyst with a keen eye for market trends.',
    tools=[web_search_tool],  # Only uses web search tool
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging blog posts about the AI industry',
    backstory='A skilled writer with a passion for technology.',
    tools=[],  # No tools needed; writes based on research summary
    verbose=True
)

from crewai import Task, Crew

# Define Tasks
research = Task(
    description='Search the web for the latest AI trends and provide a summarized report.',
    expected_output='A summary of the top 3 trending developments in AI with insights on their impact.',
    agent=researcher
)

write = Task(
    description='Write an engaging blog post about the AI industry based on the research analyst’s summary.',
    expected_output='A well-structured, 4-paragraph blog post in markdown format with simple, engaging content.',
    agent=writer,
    output_file='blog-posts/new_post.md'  # Saves blog post in 'blog-posts' directory
)

# Assemble the Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research, write],
    verbose=True,
    planning=True  # Enable AI planning feature
)

# Execute the Tasks
crew.kickoff()