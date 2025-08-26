import json
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# Initialize LLM and memory
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Global storage for application info
application_info = {"name": None, "email": None, "skills": None}

# --- LLM-based extraction ---
response_schemas = [
    ResponseSchema(name="name", description="The applicant's full name, if present."),
    ResponseSchema(name="email", description="The applicant's email address, if present."),
    ResponseSchema(name="skills", description="A list of skills mentioned."),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant that extracts job application information from text. "
        "Always return valid JSON with the following exact keys, all lowercase:\n"
        "- name (string)\n"
        "- email (string)\n"
        "- skills (array of strings)\n\n"
        "Do not change the casing of keys. Keys must be exactly as written."
    ),
    (
        "user",
        "{text}\n\nReturn only the JSON object, no explanations."
    ),
])

def extract_application_info_llm(text: str) -> str:
    prompt = prompt_template.format_messages(text=text)
    response = llm(prompt)

    try:
        parsed = output_parser.parse( response.content)
       
        
    except Exception:
        raw_json = json.loads(response.content)
        raw_json = {k.lower(): v for k, v in raw_json.items()}
        parsed = raw_json

    for key in application_info:
        if parsed.get(key):
            application_info[key] = parsed[key]

    return "Got it. Let me check what else I need."

# Extract text from uploaded CV
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_info_from_cv_llm(text: str):
    return extract_application_info_llm(text)

# Goal checker
def check_application_goal(_: str) -> str:
    if all(application_info.values()):
        return f"✅ You're ready! Name: {application_info['name']}, Email: {application_info['email']}, Skills: {application_info['skills']}."
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f"⏳ Still need: {', '.join(missing)}"

# Define tools for LangChain agent
tools = [
    Tool(
        name="extract_application_info",
        func=extract_application_info_llm,
        description="Use this to extract name, email, and skills from text."
    ),
    Tool(
        name="check_application_goal",
        func=check_application_goal,
        description="Check whether all required fields (name, email, skills) are filled."
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False
)

# --- Streamlit UI ---
st.set_page_config(page_title="🎯 Job Application Assistant", layout="centered")
st.title("🧠 Goal-Based Agent: Job Application Assistant")
st.markdown("Tell me your **name**, **email**, and **skills** to complete your application!")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "goal_complete" not in st.session_state:
    st.session_state.goal_complete = False
if "download_ready" not in st.session_state:
    st.session_state.download_ready = False
if "application_summary" not in st.session_state:
    st.session_state.application_summary = ""

# Upload resume
st.sidebar.header("📤 Upload Resume (Optional)")
resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt"])

if resume:
    st.sidebar.success("Resume uploaded!")
    text = extract_text_from_pdf(resume)
    extract_info_from_cv_llm(text)
    st.sidebar.info("🔍 Extracted info from resume:")
    for key, value in application_info.items():
        st.sidebar.markdown(f"**{key.capitalize()}:** {value}")

# Reset chat
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history.clear()
    st.session_state.goal_complete = False
    st.session_state.download_ready = False
    st.session_state.application_summary = ""
    for key in application_info:
        application_info[key] = None
    st.rerun()

# Chat input
user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    extract_application_info_llm(user_input)
    response = agent.invoke({"input": user_input})
    bot_reply = response["output"]
    st.session_state.chat_history.append(("bot", bot_reply))
    goal_status = check_application_goal("check")
    st.session_state.chat_history.append(("status", goal_status))

    if "you're ready" in goal_status.lower():
        st.session_state.goal_complete = True
        summary = (
            f"✅ Name: {application_info['name']}\n"
            f"📧 Email: {application_info['email']}\n"
            f"🛠️ Skills: {application_info['skills']}\n"
        )
        st.session_state.application_summary = summary
        st.session_state.download_ready = True

# Chat UI with avatars
for sender, message in st.session_state.chat_history:
    if sender == "user":
        with st.chat_message("🧑"):
            st.markdown(message)
    elif sender == "bot":
        with st.chat_message("🤖"):
            st.markdown(message)
    elif sender == "status":
        with st.chat_message("📊"):
            st.info(message)

# Final message
if st.session_state.goal_complete:
    st.success("🎉 All information collected! You're ready to apply!")

# Download summary
if st.session_state.download_ready:
    st.download_button(
        label="📥 Download Application Summary",
        data=st.session_state.application_summary,
        file_name="application_summary.txt",
        mime="text/plain"
    )
