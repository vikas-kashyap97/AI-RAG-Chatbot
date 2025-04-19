import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Initialize models and tools
genai_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=2)

# Create the agent
system_prompt = "You are a highly intelligent and helpful AI assistant. You provide detailed, accurate, and reliable responses. Your tone is friendly, professional, and respectful. Always consider the context of the question, ask clarifying questions if needed, and try to simplify complex topics for users when possible. Use up-to-date information when allowed."


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="Google":
        llm=ChatGoogleGenerativeAI(model=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    state = {"messages": query}
    response = agent.invoke(state)

    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

    return {"response": ai_messages[-1]}

