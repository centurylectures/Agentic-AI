
from agno.knowledge.arxiv import ArxivKnowledgeBase
from agno.vectordb.pgvector import PgVector



import os
from dotenv import load_dotenv
from agno.models.openai import OpenAIChat
from agno.agent import Agent

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env!")
os.environ["OPENAI_API_KEY"] = api_key

# Load LLM Model
llm = OpenAIChat(id="gpt-4o")

# Setup knowledge base
knowledge_base = ArxivKnowledgeBase(
    queries=["Generative AI", "Machine Learning"],
    vector_db=PgVector(
        table_name="arxiv_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5432/ai",
    ),
)

agent = Agent(
    knowledge=knowledge_base,
    search_knowledge=True,
)

# Downloads and stores papers in the vector database only if not already saved (recreate=False)
agent.knowledge.load(recreate=False)

# If you want user can ask single question then uncomment these 2 lines and comment the while loop.

# Ask user for a question
user_question = input("\n Ask a question from the knowledge base: ")

# Generate and print response
agent.print_response(user_question, user_id="user_1", stream=True)



# latest paper in generative ai
# most influential papers in generative ai

